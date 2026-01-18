import bisect
import textwrap
from collections import defaultdict
from nltk.tag import BrillTagger, untag
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