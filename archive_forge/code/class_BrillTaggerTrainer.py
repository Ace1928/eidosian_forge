import bisect
import textwrap
from collections import defaultdict
from nltk.tag import BrillTagger, untag
class BrillTaggerTrainer:
    """
    A trainer for tbl taggers.
    """

    def __init__(self, initial_tagger, templates, trace=0, deterministic=None, ruleformat='str'):
        """
        Construct a Brill tagger from a baseline tagger and a
        set of templates

        :param initial_tagger: the baseline tagger
        :type initial_tagger: Tagger
        :param templates: templates to be used in training
        :type templates: list of Templates
        :param trace: verbosity level
        :type trace: int
        :param deterministic: if True, adjudicate ties deterministically
        :type deterministic: bool
        :param ruleformat: format of reported Rules
        :type ruleformat: str
        :return: An untrained BrillTagger
        :rtype: BrillTagger
        """
        if deterministic is None:
            deterministic = trace > 0
        self._initial_tagger = initial_tagger
        self._templates = templates
        self._trace = trace
        self._deterministic = deterministic
        self._ruleformat = ruleformat
        self._tag_positions = None
        'Mapping from tags to lists of positions that use that tag.'
        self._rules_by_position = None
        "Mapping from positions to the set of rules that are known\n           to occur at that position.  Position is (sentnum, wordnum).\n           Initially, this will only contain positions where each rule\n           applies in a helpful way; but when we examine a rule, we'll\n           extend this list to also include positions where each rule\n           applies in a harmful or neutral way."
        self._positions_by_rule = None
        "Mapping from rule to position to effect, specifying the\n           effect that each rule has on the overall score, at each\n           position.  Position is (sentnum, wordnum); and effect is\n           -1, 0, or 1.  As with _rules_by_position, this mapping starts\n           out only containing rules with positive effects; but when\n           we examine a rule, we'll extend this mapping to include\n           the positions where the rule is harmful or neutral."
        self._rules_by_score = None
        'Mapping from scores to the set of rules whose effect on the\n           overall score is upper bounded by that score.  Invariant:\n           rulesByScore[s] will contain r iff the sum of\n           _positions_by_rule[r] is s.'
        self._rule_scores = None
        'Mapping from rules to upper bounds on their effects on the\n           overall score.  This is the inverse mapping to _rules_by_score.\n           Invariant: ruleScores[r] = sum(_positions_by_rule[r])'
        self._first_unknown_position = None
        "Mapping from rules to the first position where we're unsure\n           if the rule applies.  This records the next position we\n           need to check to see if the rule messed anything up."

    def train(self, train_sents, max_rules=200, min_score=2, min_acc=None):
        """
        Trains the Brill tagger on the corpus *train_sents*,
        producing at most *max_rules* transformations, each of which
        reduces the net number of errors in the corpus by at least
        *min_score*, and each of which has accuracy not lower than
        *min_acc*.

        >>> # Relevant imports
        >>> from nltk.tbl.template import Template
        >>> from nltk.tag.brill import Pos, Word
        >>> from nltk.tag import untag, RegexpTagger, BrillTaggerTrainer

        >>> # Load some data
        >>> from nltk.corpus import treebank
        >>> training_data = treebank.tagged_sents()[:100]
        >>> baseline_data = treebank.tagged_sents()[100:200]
        >>> gold_data = treebank.tagged_sents()[200:300]
        >>> testing_data = [untag(s) for s in gold_data]

        >>> backoff = RegexpTagger([
        ... (r'^-?[0-9]+(\\.[0-9]+)?$', 'CD'),  # cardinal numbers
        ... (r'(The|the|A|a|An|an)$', 'AT'),   # articles
        ... (r'.*able$', 'JJ'),                # adjectives
        ... (r'.*ness$', 'NN'),                # nouns formed from adjectives
        ... (r'.*ly$', 'RB'),                  # adverbs
        ... (r'.*s$', 'NNS'),                  # plural nouns
        ... (r'.*ing$', 'VBG'),                # gerunds
        ... (r'.*ed$', 'VBD'),                 # past tense verbs
        ... (r'.*', 'NN')                      # nouns (default)
        ... ])

        >>> baseline = backoff #see NOTE1
        >>> baseline.accuracy(gold_data) #doctest: +ELLIPSIS
        0.243...

        >>> # Set up templates
        >>> Template._cleartemplates() #clear any templates created in earlier tests
        >>> templates = [Template(Pos([-1])), Template(Pos([-1]), Word([0]))]

        >>> # Construct a BrillTaggerTrainer
        >>> tt = BrillTaggerTrainer(baseline, templates, trace=3)

        >>> tagger1 = tt.train(training_data, max_rules=10)
        TBL train (fast) (seqs: 100; tokens: 2417; tpls: 2; min score: 2; min acc: None)
        Finding initial useful rules...
            Found 847 useful rules.
        <BLANKLINE>
                   B      |
           S   F   r   O  |        Score = Fixed - Broken
           c   i   o   t  |  R     Fixed = num tags changed incorrect -> correct
           o   x   k   h  |  u     Broken = num tags changed correct -> incorrect
           r   e   e   e  |  l     Other = num tags changed incorrect -> incorrect
           e   d   n   r  |  e
        ------------------+-------------------------------------------------------
         132 132   0   0  | AT->DT if Pos:NN@[-1]
          85  85   0   0  | NN->, if Pos:NN@[-1] & Word:,@[0]
          69  69   0   0  | NN->. if Pos:NN@[-1] & Word:.@[0]
          51  51   0   0  | NN->IN if Pos:NN@[-1] & Word:of@[0]
          47  63  16 162  | NN->IN if Pos:NNS@[-1]
          33  33   0   0  | NN->TO if Pos:NN@[-1] & Word:to@[0]
          26  26   0   0  | IN->. if Pos:NNS@[-1] & Word:.@[0]
          24  24   0   0  | IN->, if Pos:NNS@[-1] & Word:,@[0]
          22  27   5  24  | NN->-NONE- if Pos:VBD@[-1]
          17  17   0   0  | NN->CC if Pos:NN@[-1] & Word:and@[0]

        >>> tagger1.rules()[1:3]
        (Rule('001', 'NN', ',', [(Pos([-1]),'NN'), (Word([0]),',')]), Rule('001', 'NN', '.', [(Pos([-1]),'NN'), (Word([0]),'.')]))

        >>> train_stats = tagger1.train_stats()
        >>> [train_stats[stat] for stat in ['initialerrors', 'finalerrors', 'rulescores']]
        [1776, 1270, [132, 85, 69, 51, 47, 33, 26, 24, 22, 17]]

        >>> tagger1.print_template_statistics(printunused=False)
        TEMPLATE STATISTICS (TRAIN)  2 templates, 10 rules)
        TRAIN (   2417 tokens) initial  1776 0.2652 final:  1270 0.4746
        #ID | Score (train) |  #Rules     | Template
        --------------------------------------------
        001 |   305   0.603 |   7   0.700 | Template(Pos([-1]),Word([0]))
        000 |   201   0.397 |   3   0.300 | Template(Pos([-1]))
        <BLANKLINE>
        <BLANKLINE>

        >>> round(tagger1.accuracy(gold_data),5)
        0.43834

        >>> tagged, test_stats = tagger1.batch_tag_incremental(testing_data, gold_data)

        >>> tagged[33][12:] == [('foreign', 'IN'), ('debt', 'NN'), ('of', 'IN'), ('$', 'NN'), ('64', 'CD'),
        ... ('billion', 'NN'), ('*U*', 'NN'), ('--', 'NN'), ('the', 'DT'), ('third-highest', 'NN'), ('in', 'NN'),
        ... ('the', 'DT'), ('developing', 'VBG'), ('world', 'NN'), ('.', '.')]
        True

        >>> [test_stats[stat] for stat in ['initialerrors', 'finalerrors', 'rulescores']]
        [1859, 1380, [100, 85, 67, 58, 27, 36, 27, 16, 31, 32]]

        >>> # A high-accuracy tagger
        >>> tagger2 = tt.train(training_data, max_rules=10, min_acc=0.99)
        TBL train (fast) (seqs: 100; tokens: 2417; tpls: 2; min score: 2; min acc: 0.99)
        Finding initial useful rules...
            Found 847 useful rules.
        <BLANKLINE>
                   B      |
           S   F   r   O  |        Score = Fixed - Broken
           c   i   o   t  |  R     Fixed = num tags changed incorrect -> correct
           o   x   k   h  |  u     Broken = num tags changed correct -> incorrect
           r   e   e   e  |  l     Other = num tags changed incorrect -> incorrect
           e   d   n   r  |  e
        ------------------+-------------------------------------------------------
         132 132   0   0  | AT->DT if Pos:NN@[-1]
          85  85   0   0  | NN->, if Pos:NN@[-1] & Word:,@[0]
          69  69   0   0  | NN->. if Pos:NN@[-1] & Word:.@[0]
          51  51   0   0  | NN->IN if Pos:NN@[-1] & Word:of@[0]
          36  36   0   0  | NN->TO if Pos:NN@[-1] & Word:to@[0]
          26  26   0   0  | NN->. if Pos:NNS@[-1] & Word:.@[0]
          24  24   0   0  | NN->, if Pos:NNS@[-1] & Word:,@[0]
          19  19   0   6  | NN->VB if Pos:TO@[-1]
          18  18   0   0  | CD->-NONE- if Pos:NN@[-1] & Word:0@[0]
          18  18   0   0  | NN->CC if Pos:NN@[-1] & Word:and@[0]

        >>> round(tagger2.accuracy(gold_data), 8)
        0.43996744

        >>> tagger2.rules()[2:4]
        (Rule('001', 'NN', '.', [(Pos([-1]),'NN'), (Word([0]),'.')]), Rule('001', 'NN', 'IN', [(Pos([-1]),'NN'), (Word([0]),'of')]))

        # NOTE1: (!!FIXME) A far better baseline uses nltk.tag.UnigramTagger,
        # with a RegexpTagger only as backoff. For instance,
        # >>> baseline = UnigramTagger(baseline_data, backoff=backoff)
        # However, as of Nov 2013, nltk.tag.UnigramTagger does not yield consistent results
        # between python versions. The simplistic backoff above is a workaround to make doctests
        # get consistent input.

        :param train_sents: training data
        :type train_sents: list(list(tuple))
        :param max_rules: output at most max_rules rules
        :type max_rules: int
        :param min_score: stop training when no rules better than min_score can be found
        :type min_score: int
        :param min_acc: discard any rule with lower accuracy than min_acc
        :type min_acc: float or None
        :return: the learned tagger
        :rtype: BrillTagger
        """
        test_sents = [list(self._initial_tagger.tag(untag(sent))) for sent in train_sents]
        trainstats = {}
        trainstats['min_acc'] = min_acc
        trainstats['min_score'] = min_score
        trainstats['tokencount'] = sum((len(t) for t in test_sents))
        trainstats['sequencecount'] = len(test_sents)
        trainstats['templatecount'] = len(self._templates)
        trainstats['rulescores'] = []
        trainstats['initialerrors'] = sum((tag[1] != truth[1] for paired in zip(test_sents, train_sents) for tag, truth in zip(*paired)))
        trainstats['initialacc'] = 1 - trainstats['initialerrors'] / trainstats['tokencount']
        if self._trace > 0:
            print('TBL train (fast) (seqs: {sequencecount}; tokens: {tokencount}; tpls: {templatecount}; min score: {min_score}; min acc: {min_acc})'.format(**trainstats))
        if self._trace:
            print('Finding initial useful rules...')
        self._init_mappings(test_sents, train_sents)
        if self._trace:
            print(f'    Found {len(self._rule_scores)} useful rules.')
        if self._trace > 2:
            self._trace_header()
        elif self._trace == 1:
            print('Selecting rules...')
        rules = []
        try:
            while len(rules) < max_rules:
                rule = self._best_rule(train_sents, test_sents, min_score, min_acc)
                if rule:
                    rules.append(rule)
                    score = self._rule_scores[rule]
                    trainstats['rulescores'].append(score)
                else:
                    break
                if self._trace > 1:
                    self._trace_rule(rule)
                self._apply_rule(rule, test_sents)
                self._update_tag_positions(rule)
                self._update_rules(rule, train_sents, test_sents)
        except KeyboardInterrupt:
            print(f'Training stopped manually -- {len(rules)} rules found')
        self._clean()
        trainstats['finalerrors'] = trainstats['initialerrors'] - sum(trainstats['rulescores'])
        trainstats['finalacc'] = 1 - trainstats['finalerrors'] / trainstats['tokencount']
        return BrillTagger(self._initial_tagger, rules, trainstats)

    def _init_mappings(self, test_sents, train_sents):
        """
        Initialize the tag position mapping & the rule related
        mappings.  For each error in test_sents, find new rules that
        would correct them, and add them to the rule mappings.
        """
        self._tag_positions = defaultdict(list)
        self._rules_by_position = defaultdict(set)
        self._positions_by_rule = defaultdict(dict)
        self._rules_by_score = defaultdict(set)
        self._rule_scores = defaultdict(int)
        self._first_unknown_position = defaultdict(int)
        for sentnum, sent in enumerate(test_sents):
            for wordnum, (word, tag) in enumerate(sent):
                self._tag_positions[tag].append((sentnum, wordnum))
                correct_tag = train_sents[sentnum][wordnum][1]
                if tag != correct_tag:
                    for rule in self._find_rules(sent, wordnum, correct_tag):
                        self._update_rule_applies(rule, sentnum, wordnum, train_sents)

    def _clean(self):
        self._tag_positions = None
        self._rules_by_position = None
        self._positions_by_rule = None
        self._rules_by_score = None
        self._rule_scores = None
        self._first_unknown_position = None

    def _find_rules(self, sent, wordnum, new_tag):
        """
        Use the templates to find rules that apply at index *wordnum*
        in the sentence *sent* and generate the tag *new_tag*.
        """
        for template in self._templates:
            yield from template.applicable_rules(sent, wordnum, new_tag)

    def _update_rule_applies(self, rule, sentnum, wordnum, train_sents):
        """
        Update the rule data tables to reflect the fact that
        *rule* applies at the position *(sentnum, wordnum)*.
        """
        pos = (sentnum, wordnum)
        if pos in self._positions_by_rule[rule]:
            return
        correct_tag = train_sents[sentnum][wordnum][1]
        if rule.replacement_tag == correct_tag:
            self._positions_by_rule[rule][pos] = 1
        elif rule.original_tag == correct_tag:
            self._positions_by_rule[rule][pos] = -1
        else:
            self._positions_by_rule[rule][pos] = 0
        self._rules_by_position[pos].add(rule)
        old_score = self._rule_scores[rule]
        self._rule_scores[rule] += self._positions_by_rule[rule][pos]
        self._rules_by_score[old_score].discard(rule)
        self._rules_by_score[self._rule_scores[rule]].add(rule)

    def _update_rule_not_applies(self, rule, sentnum, wordnum):
        """
        Update the rule data tables to reflect the fact that *rule*
        does not apply at the position *(sentnum, wordnum)*.
        """
        pos = (sentnum, wordnum)
        old_score = self._rule_scores[rule]
        self._rule_scores[rule] -= self._positions_by_rule[rule][pos]
        self._rules_by_score[old_score].discard(rule)
        self._rules_by_score[self._rule_scores[rule]].add(rule)
        del self._positions_by_rule[rule][pos]
        self._rules_by_position[pos].remove(rule)

    def _best_rule(self, train_sents, test_sents, min_score, min_acc):
        """
        Find the next best rule.  This is done by repeatedly taking a
        rule with the highest score and stepping through the corpus to
        see where it applies.  When it makes an error (decreasing its
        score) it's bumped down, and we try a new rule with the
        highest score.  When we find a rule which has the highest
        score *and* which has been tested against the entire corpus, we
        can conclude that it's the next best rule.
        """
        for max_score in sorted(self._rules_by_score.keys(), reverse=True):
            if len(self._rules_by_score) == 0:
                return None
            if max_score < min_score or max_score <= 0:
                return None
            best_rules = list(self._rules_by_score[max_score])
            if self._deterministic:
                best_rules.sort(key=repr)
            for rule in best_rules:
                positions = self._tag_positions[rule.original_tag]
                unk = self._first_unknown_position.get(rule, (0, -1))
                start = bisect.bisect_left(positions, unk)
                for i in range(start, len(positions)):
                    sentnum, wordnum = positions[i]
                    if rule.applies(test_sents[sentnum], wordnum):
                        self._update_rule_applies(rule, sentnum, wordnum, train_sents)
                        if self._rule_scores[rule] < max_score:
                            self._first_unknown_position[rule] = (sentnum, wordnum + 1)
                            break
                if self._rule_scores[rule] == max_score:
                    self._first_unknown_position[rule] = (len(train_sents) + 1, 0)
                    if min_acc is None:
                        return rule
                    else:
                        changes = self._positions_by_rule[rule].values()
                        num_fixed = len([c for c in changes if c == 1])
                        num_broken = len([c for c in changes if c == -1])
                        acc = num_fixed / (num_fixed + num_broken)
                        if acc >= min_acc:
                            return rule
            assert min_acc is not None or not self._rules_by_score[max_score]
            if not self._rules_by_score[max_score]:
                del self._rules_by_score[max_score]

    def _apply_rule(self, rule, test_sents):
        """
        Update *test_sents* by applying *rule* everywhere where its
        conditions are met.
        """
        update_positions = set(self._positions_by_rule[rule])
        new_tag = rule.replacement_tag
        if self._trace > 3:
            self._trace_apply(len(update_positions))
        for sentnum, wordnum in update_positions:
            text = test_sents[sentnum][wordnum][0]
            test_sents[sentnum][wordnum] = (text, new_tag)

    def _update_tag_positions(self, rule):
        """
        Update _tag_positions to reflect the changes to tags that are
        made by *rule*.
        """
        for pos in self._positions_by_rule[rule]:
            old_tag_positions = self._tag_positions[rule.original_tag]
            old_index = bisect.bisect_left(old_tag_positions, pos)
            del old_tag_positions[old_index]
            new_tag_positions = self._tag_positions[rule.replacement_tag]
            bisect.insort_left(new_tag_positions, pos)

    def _update_rules(self, rule, train_sents, test_sents):
        """
        Check if we should add or remove any rules from consideration,
        given the changes made by *rule*.
        """
        neighbors = set()
        for sentnum, wordnum in self._positions_by_rule[rule]:
            for template in self._templates:
                n = template.get_neighborhood(test_sents[sentnum], wordnum)
                neighbors.update([(sentnum, i) for i in n])
        num_obsolete = num_new = num_unseen = 0
        for sentnum, wordnum in neighbors:
            test_sent = test_sents[sentnum]
            correct_tag = train_sents[sentnum][wordnum][1]
            old_rules = set(self._rules_by_position[sentnum, wordnum])
            for old_rule in old_rules:
                if not old_rule.applies(test_sent, wordnum):
                    num_obsolete += 1
                    self._update_rule_not_applies(old_rule, sentnum, wordnum)
            for template in self._templates:
                for new_rule in template.applicable_rules(test_sent, wordnum, correct_tag):
                    if new_rule not in old_rules:
                        num_new += 1
                        if new_rule not in self._rule_scores:
                            num_unseen += 1
                        old_rules.add(new_rule)
                        self._update_rule_applies(new_rule, sentnum, wordnum, train_sents)
            for new_rule, pos in self._first_unknown_position.items():
                if pos > (sentnum, wordnum):
                    if new_rule not in old_rules:
                        num_new += 1
                        if new_rule.applies(test_sent, wordnum):
                            self._update_rule_applies(new_rule, sentnum, wordnum, train_sents)
        if self._trace > 3:
            self._trace_update_rules(num_obsolete, num_new, num_unseen)

    def _trace_header(self):
        print('\n           B      |\n   S   F   r   O  |        Score = Fixed - Broken\n   c   i   o   t  |  R     Fixed = num tags changed incorrect -> correct\n   o   x   k   h  |  u     Broken = num tags changed correct -> incorrect\n   r   e   e   e  |  l     Other = num tags changed incorrect -> incorrect\n   e   d   n   r  |  e\n------------------+-------------------------------------------------------\n        '.rstrip())

    def _trace_rule(self, rule):
        assert self._rule_scores[rule] == sum(self._positions_by_rule[rule].values())
        changes = self._positions_by_rule[rule].values()
        num_fixed = len([c for c in changes if c == 1])
        num_broken = len([c for c in changes if c == -1])
        num_other = len([c for c in changes if c == 0])
        score = self._rule_scores[rule]
        rulestr = rule.format(self._ruleformat)
        if self._trace > 2:
            print('{:4d}{:4d}{:4d}{:4d}  |'.format(score, num_fixed, num_broken, num_other), end=' ')
            print(textwrap.fill(rulestr, initial_indent=' ' * 20, width=79, subsequent_indent=' ' * 18 + '|   ').strip())
        else:
            print(rulestr)

    def _trace_apply(self, num_updates):
        prefix = ' ' * 18 + '|'
        print(prefix)
        print(prefix, f'Applying rule to {num_updates} positions.')

    def _trace_update_rules(self, num_obsolete, num_new, num_unseen):
        prefix = ' ' * 18 + '|'
        print(prefix, 'Updated rule tables:')
        print(prefix, f'  - {num_obsolete} rule applications removed')
        print(prefix, f'  - {num_new} rule applications added ({num_unseen} novel)')
        print(prefix)