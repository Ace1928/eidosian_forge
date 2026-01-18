from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.SemanticContext import Predicate, PrecedencePredicate
from antlr4.atn.ATNState import *
class WildcardTransition(Transition):
    __slots__ = 'serializationType'

    def __init__(self, target: ATNState):
        super().__init__(target)
        self.serializationType = self.WILDCARD

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        return symbol >= minVocabSymbol and symbol <= maxVocabSymbol

    def __str__(self):
        return '.'