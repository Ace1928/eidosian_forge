import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
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