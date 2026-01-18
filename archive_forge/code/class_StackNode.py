from typing import Dict, Type
from parso import tree
from parso.pgen2.generator import ReservedString
class StackNode:

    def __init__(self, dfa):
        self.dfa = dfa
        self.nodes = []

    @property
    def nonterminal(self):
        return self.dfa.from_rule

    def __repr__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, self.dfa, self.nodes)