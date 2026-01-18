import bisect
import textwrap
from collections import defaultdict
from nltk.tag import BrillTagger, untag
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