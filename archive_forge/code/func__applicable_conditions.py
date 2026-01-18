import itertools as it
from abc import ABCMeta, abstractmethod
from nltk.tbl.feature import Feature
from nltk.tbl.rule import Rule
def _applicable_conditions(self, tokens, index):
    """
        :returns: A set of all conditions for rules
        that are applicable to C{tokens[index]}.
        """
    conditions = []
    for feature in self._features:
        conditions.append([])
        for pos in feature.positions:
            if not 0 <= index + pos < len(tokens):
                continue
            value = feature.extract_property(tokens, index + pos)
            conditions[-1].append((feature, value))
    return conditions