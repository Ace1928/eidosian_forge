from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.SemanticContext import Predicate, PrecedencePredicate
from antlr4.atn.ATNState import *
class PrecedencePredicateTransition(AbstractPredicateTransition):
    __slots__ = ('serializationType', 'precedence')

    def __init__(self, target: ATNState, precedence: int):
        super().__init__(target)
        self.serializationType = self.PRECEDENCE
        self.precedence = precedence
        self.isEpsilon = True

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        return False

    def getPredicate(self):
        return PrecedencePredicate(self.precedence)

    def __str__(self):
        return self.precedence + ' >= _p'