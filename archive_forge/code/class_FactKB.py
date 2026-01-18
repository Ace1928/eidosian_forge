from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
class FactKB(dict):
    """
    A simple propositional knowledge base relying on compiled inference rules.
    """

    def __str__(self):
        return '{\n%s}' % ',\n'.join(['\t%s: %s' % i for i in sorted(self.items())])

    def __init__(self, rules):
        self.rules = rules

    def _tell(self, k, v):
        """Add fact k=v to the knowledge base.

        Returns True if the KB has actually been updated, False otherwise.
        """
        if k in self and self[k] is not None:
            if self[k] == v:
                return False
            else:
                raise InconsistentAssumptions(self, k, v)
        else:
            self[k] = v
            return True

    def deduce_all_facts(self, facts):
        """
        Update the KB with all the implications of a list of facts.

        Facts can be specified as a dictionary or as a list of (key, value)
        pairs.
        """
        full_implications = self.rules.full_implications
        beta_triggers = self.rules.beta_triggers
        beta_rules = self.rules.beta_rules
        if isinstance(facts, dict):
            facts = facts.items()
        while facts:
            beta_maytrigger = set()
            for k, v in facts:
                if not self._tell(k, v) or v is None:
                    continue
                for key, value in full_implications[k, v]:
                    self._tell(key, value)
                beta_maytrigger.update(beta_triggers[k, v])
            facts = []
            for bidx in beta_maytrigger:
                bcond, bimpl = beta_rules[bidx]
                if all((self.get(k) is v for k, v in bcond)):
                    facts.append(bimpl)