import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
class PunktTrainer(PunktBaseClass):
    """Learns parameters used in Punkt sentence boundary detection."""

    def __init__(self, train_text=None, verbose=False, lang_vars=None, token_cls=PunktToken):
        PunktBaseClass.__init__(self, lang_vars=lang_vars, token_cls=token_cls)
        self._type_fdist = FreqDist()
        'A frequency distribution giving the frequency of each\n        case-normalized token type in the training data.'
        self._num_period_toks = 0
        'The number of words ending in period in the training data.'
        self._collocation_fdist = FreqDist()
        'A frequency distribution giving the frequency of all\n        bigrams in the training data where the first word ends in a\n        period.  Bigrams are encoded as tuples of word types.\n        Especially common collocations are extracted from this\n        frequency distribution, and stored in\n        ``_params``.``collocations <PunktParameters.collocations>``.'
        self._sent_starter_fdist = FreqDist()
        'A frequency distribution giving the frequency of all words\n        that occur at the training data at the beginning of a sentence\n        (after the first pass of annotation).  Especially common\n        sentence starters are extracted from this frequency\n        distribution, and stored in ``_params.sent_starters``.\n        '
        self._sentbreak_count = 0
        'The total number of sentence breaks identified in training, used for\n        calculating the frequent sentence starter heuristic.'
        self._finalized = True
        'A flag as to whether the training has been finalized by finding\n        collocations and sentence starters, or whether finalize_training()\n        still needs to be called.'
        if train_text:
            self.train(train_text, verbose, finalize=True)

    def get_params(self):
        """
        Calculates and returns parameters for sentence boundary detection as
        derived from training."""
        if not self._finalized:
            self.finalize_training()
        return self._params
    ABBREV = 0.3
    "cut-off value whether a 'token' is an abbreviation"
    IGNORE_ABBREV_PENALTY = False
    'allows the disabling of the abbreviation penalty heuristic, which\n    exponentially disadvantages words that are found at times without a\n    final period.'
    ABBREV_BACKOFF = 5
    "upper cut-off for Mikheev's(2002) abbreviation detection algorithm"
    COLLOCATION = 7.88
    'minimal log-likelihood value that two tokens need to be considered\n    as a collocation'
    SENT_STARTER = 30
    'minimal log-likelihood value that a token requires to be considered\n    as a frequent sentence starter'
    INCLUDE_ALL_COLLOCS = False
    'this includes as potential collocations all word pairs where the first\n    word ends in a period. It may be useful in corpora where there is a lot\n    of variation that makes abbreviations like Mr difficult to identify.'
    INCLUDE_ABBREV_COLLOCS = False
    'this includes as potential collocations all word pairs where the first\n    word is an abbreviation. Such collocations override the orthographic\n    heuristic, but not the sentence starter heuristic. This is overridden by\n    INCLUDE_ALL_COLLOCS, and if both are false, only collocations with initials\n    and ordinals are considered.'
    ''
    MIN_COLLOC_FREQ = 1
    'this sets a minimum bound on the number of times a bigram needs to\n    appear before it can be considered a collocation, in addition to log\n    likelihood statistics. This is useful when INCLUDE_ALL_COLLOCS is True.'

    def train(self, text, verbose=False, finalize=True):
        """
        Collects training data from a given text. If finalize is True, it
        will determine all the parameters for sentence boundary detection. If
        not, this will be delayed until get_params() or finalize_training() is
        called. If verbose is True, abbreviations found will be listed.
        """
        self._train_tokens(self._tokenize_words(text), verbose)
        if finalize:
            self.finalize_training(verbose)

    def train_tokens(self, tokens, verbose=False, finalize=True):
        """
        Collects training data from a given list of tokens.
        """
        self._train_tokens((self._Token(t) for t in tokens), verbose)
        if finalize:
            self.finalize_training(verbose)

    def _train_tokens(self, tokens, verbose):
        self._finalized = False
        tokens = list(tokens)
        for aug_tok in tokens:
            self._type_fdist[aug_tok.type] += 1
            if aug_tok.period_final:
                self._num_period_toks += 1
        unique_types = self._unique_types(tokens)
        for abbr, score, is_add in self._reclassify_abbrev_types(unique_types):
            if score >= self.ABBREV:
                if is_add:
                    self._params.abbrev_types.add(abbr)
                    if verbose:
                        print(f'  Abbreviation: [{score:6.4f}] {abbr}')
            elif not is_add:
                self._params.abbrev_types.remove(abbr)
                if verbose:
                    print(f'  Removed abbreviation: [{score:6.4f}] {abbr}')
        tokens = list(self._annotate_first_pass(tokens))
        self._get_orthography_data(tokens)
        self._sentbreak_count += self._get_sentbreak_count(tokens)
        for aug_tok1, aug_tok2 in _pair_iter(tokens):
            if not aug_tok1.period_final or not aug_tok2:
                continue
            if self._is_rare_abbrev_type(aug_tok1, aug_tok2):
                self._params.abbrev_types.add(aug_tok1.type_no_period)
                if verbose:
                    print('  Rare Abbrev: %s' % aug_tok1.type)
            if self._is_potential_sent_starter(aug_tok2, aug_tok1):
                self._sent_starter_fdist[aug_tok2.type] += 1
            if self._is_potential_collocation(aug_tok1, aug_tok2):
                self._collocation_fdist[aug_tok1.type_no_period, aug_tok2.type_no_sentperiod] += 1

    def _unique_types(self, tokens):
        return {aug_tok.type for aug_tok in tokens}

    def finalize_training(self, verbose=False):
        """
        Uses data that has been gathered in training to determine likely
        collocations and sentence starters.
        """
        self._params.clear_sent_starters()
        for typ, log_likelihood in self._find_sent_starters():
            self._params.sent_starters.add(typ)
            if verbose:
                print(f'  Sent Starter: [{log_likelihood:6.4f}] {typ!r}')
        self._params.clear_collocations()
        for (typ1, typ2), log_likelihood in self._find_collocations():
            self._params.collocations.add((typ1, typ2))
            if verbose:
                print(f'  Collocation: [{log_likelihood:6.4f}] {typ1!r}+{typ2!r}')
        self._finalized = True

    def freq_threshold(self, ortho_thresh=2, type_thresh=2, colloc_thres=2, sentstart_thresh=2):
        """
        Allows memory use to be reduced after much training by removing data
        about rare tokens that are unlikely to have a statistical effect with
        further training. Entries occurring above the given thresholds will be
        retained.
        """
        if ortho_thresh > 1:
            old_oc = self._params.ortho_context
            self._params.clear_ortho_context()
            for tok in self._type_fdist:
                count = self._type_fdist[tok]
                if count >= ortho_thresh:
                    self._params.ortho_context[tok] = old_oc[tok]
        self._type_fdist = self._freq_threshold(self._type_fdist, type_thresh)
        self._collocation_fdist = self._freq_threshold(self._collocation_fdist, colloc_thres)
        self._sent_starter_fdist = self._freq_threshold(self._sent_starter_fdist, sentstart_thresh)

    def _freq_threshold(self, fdist, threshold):
        """
        Returns a FreqDist containing only data with counts below a given
        threshold, as well as a mapping (None -> count_removed).
        """
        res = FreqDist()
        num_removed = 0
        for tok in fdist:
            count = fdist[tok]
            if count < threshold:
                num_removed += 1
            else:
                res[tok] += count
        res[None] += num_removed
        return res

    def _get_orthography_data(self, tokens):
        """
        Collect information about whether each token type occurs
        with different case patterns (i) overall, (ii) at
        sentence-initial positions, and (iii) at sentence-internal
        positions.
        """
        context = 'internal'
        tokens = list(tokens)
        for aug_tok in tokens:
            if aug_tok.parastart and context != 'unknown':
                context = 'initial'
            if aug_tok.linestart and context == 'internal':
                context = 'unknown'
            typ = aug_tok.type_no_sentperiod
            flag = _ORTHO_MAP.get((context, aug_tok.first_case), 0)
            if flag:
                self._params.add_ortho_context(typ, flag)
            if aug_tok.sentbreak:
                if not (aug_tok.is_number or aug_tok.is_initial):
                    context = 'initial'
                else:
                    context = 'unknown'
            elif aug_tok.ellipsis or aug_tok.abbr:
                context = 'unknown'
            else:
                context = 'internal'

    def _reclassify_abbrev_types(self, types):
        """
        (Re)classifies each given token if
          - it is period-final and not a known abbreviation; or
          - it is not period-final and is otherwise a known abbreviation
        by checking whether its previous classification still holds according
        to the heuristics of section 3.
        Yields triples (abbr, score, is_add) where abbr is the type in question,
        score is its log-likelihood with penalties applied, and is_add specifies
        whether the present type is a candidate for inclusion or exclusion as an
        abbreviation, such that:
          - (is_add and score >= 0.3)    suggests a new abbreviation; and
          - (not is_add and score < 0.3) suggests excluding an abbreviation.
        """
        for typ in types:
            if not _re_non_punct.search(typ) or typ == '##number##':
                continue
            if typ.endswith('.'):
                if typ in self._params.abbrev_types:
                    continue
                typ = typ[:-1]
                is_add = True
            else:
                if typ not in self._params.abbrev_types:
                    continue
                is_add = False
            num_periods = typ.count('.') + 1
            num_nonperiods = len(typ) - num_periods + 1
            count_with_period = self._type_fdist[typ + '.']
            count_without_period = self._type_fdist[typ]
            log_likelihood = self._dunning_log_likelihood(count_with_period + count_without_period, self._num_period_toks, count_with_period, self._type_fdist.N())
            f_length = math.exp(-num_nonperiods)
            f_periods = num_periods
            f_penalty = int(self.IGNORE_ABBREV_PENALTY) or math.pow(num_nonperiods, -count_without_period)
            score = log_likelihood * f_length * f_periods * f_penalty
            yield (typ, score, is_add)

    def find_abbrev_types(self):
        """
        Recalculates abbreviations given type frequencies, despite no prior
        determination of abbreviations.
        This fails to include abbreviations otherwise found as "rare".
        """
        self._params.clear_abbrevs()
        tokens = (typ for typ in self._type_fdist if typ and typ.endswith('.'))
        for abbr, score, _is_add in self._reclassify_abbrev_types(tokens):
            if score >= self.ABBREV:
                self._params.abbrev_types.add(abbr)

    def _is_rare_abbrev_type(self, cur_tok, next_tok):
        """
        A word type is counted as a rare abbreviation if...
          - it's not already marked as an abbreviation
          - it occurs fewer than ABBREV_BACKOFF times
          - either it is followed by a sentence-internal punctuation
            mark, *or* it is followed by a lower-case word that
            sometimes appears with upper case, but never occurs with
            lower case at the beginning of sentences.
        """
        if cur_tok.abbr or not cur_tok.sentbreak:
            return False
        typ = cur_tok.type_no_sentperiod
        count = self._type_fdist[typ] + self._type_fdist[typ[:-1]]
        if typ in self._params.abbrev_types or count >= self.ABBREV_BACKOFF:
            return False
        if next_tok.tok[:1] in self._lang_vars.internal_punctuation:
            return True
        if next_tok.first_lower:
            typ2 = next_tok.type_no_sentperiod
            typ2ortho_context = self._params.ortho_context[typ2]
            if typ2ortho_context & _ORTHO_BEG_UC and (not typ2ortho_context & _ORTHO_MID_UC):
                return True

    @staticmethod
    def _dunning_log_likelihood(count_a, count_b, count_ab, N):
        """
        A function that calculates the modified Dunning log-likelihood
        ratio scores for abbreviation candidates.  The details of how
        this works is available in the paper.
        """
        p1 = count_b / N
        p2 = 0.99
        null_hypo = count_ab * math.log(p1) + (count_a - count_ab) * math.log(1.0 - p1)
        alt_hypo = count_ab * math.log(p2) + (count_a - count_ab) * math.log(1.0 - p2)
        likelihood = null_hypo - alt_hypo
        return -2.0 * likelihood

    @staticmethod
    def _col_log_likelihood(count_a, count_b, count_ab, N):
        """
        A function that will just compute log-likelihood estimate, in
        the original paper it's described in algorithm 6 and 7.

        This *should* be the original Dunning log-likelihood values,
        unlike the previous log_l function where it used modified
        Dunning log-likelihood values
        """
        p = count_b / N
        p1 = count_ab / count_a
        try:
            p2 = (count_b - count_ab) / (N - count_a)
        except ZeroDivisionError:
            p2 = 1
        try:
            summand1 = count_ab * math.log(p) + (count_a - count_ab) * math.log(1.0 - p)
        except ValueError:
            summand1 = 0
        try:
            summand2 = (count_b - count_ab) * math.log(p) + (N - count_a - count_b + count_ab) * math.log(1.0 - p)
        except ValueError:
            summand2 = 0
        if count_a == count_ab or p1 <= 0 or p1 >= 1:
            summand3 = 0
        else:
            summand3 = count_ab * math.log(p1) + (count_a - count_ab) * math.log(1.0 - p1)
        if count_b == count_ab or p2 <= 0 or p2 >= 1:
            summand4 = 0
        else:
            summand4 = (count_b - count_ab) * math.log(p2) + (N - count_a - count_b + count_ab) * math.log(1.0 - p2)
        likelihood = summand1 + summand2 - summand3 - summand4
        return -2.0 * likelihood

    def _is_potential_collocation(self, aug_tok1, aug_tok2):
        """
        Returns True if the pair of tokens may form a collocation given
        log-likelihood statistics.
        """
        return (self.INCLUDE_ALL_COLLOCS or (self.INCLUDE_ABBREV_COLLOCS and aug_tok1.abbr) or (aug_tok1.sentbreak and (aug_tok1.is_number or aug_tok1.is_initial))) and aug_tok1.is_non_punct and aug_tok2.is_non_punct

    def _find_collocations(self):
        """
        Generates likely collocations and their log-likelihood.
        """
        for types in self._collocation_fdist:
            try:
                typ1, typ2 = types
            except TypeError:
                continue
            if typ2 in self._params.sent_starters:
                continue
            col_count = self._collocation_fdist[types]
            typ1_count = self._type_fdist[typ1] + self._type_fdist[typ1 + '.']
            typ2_count = self._type_fdist[typ2] + self._type_fdist[typ2 + '.']
            if typ1_count > 1 and typ2_count > 1 and (self.MIN_COLLOC_FREQ < col_count <= min(typ1_count, typ2_count)):
                log_likelihood = self._col_log_likelihood(typ1_count, typ2_count, col_count, self._type_fdist.N())
                if log_likelihood >= self.COLLOCATION and self._type_fdist.N() / typ1_count > typ2_count / col_count:
                    yield ((typ1, typ2), log_likelihood)

    def _is_potential_sent_starter(self, cur_tok, prev_tok):
        """
        Returns True given a token and the token that precedes it if it
        seems clear that the token is beginning a sentence.
        """
        return prev_tok.sentbreak and (not (prev_tok.is_number or prev_tok.is_initial)) and cur_tok.is_alpha

    def _find_sent_starters(self):
        """
        Uses collocation heuristics for each candidate token to
        determine if it frequently starts sentences.
        """
        for typ in self._sent_starter_fdist:
            if not typ:
                continue
            typ_at_break_count = self._sent_starter_fdist[typ]
            typ_count = self._type_fdist[typ] + self._type_fdist[typ + '.']
            if typ_count < typ_at_break_count:
                continue
            log_likelihood = self._col_log_likelihood(self._sentbreak_count, typ_count, typ_at_break_count, self._type_fdist.N())
            if log_likelihood >= self.SENT_STARTER and self._type_fdist.N() / self._sentbreak_count > typ_count / typ_at_break_count:
                yield (typ, log_likelihood)

    def _get_sentbreak_count(self, tokens):
        """
        Returns the number of sentence breaks marked in a given set of
        augmented tokens.
        """
        return sum((1 for aug_tok in tokens if aug_tok.sentbreak))