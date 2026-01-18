from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.SemanticContext import Predicate, PrecedencePredicate
from antlr4.atn.ATNState import *
class NotSetTransition(SetTransition):

    def __init__(self, target: ATNState, set: IntervalSet):
        super().__init__(target, set)
        self.serializationType = self.NOT_SET

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        return symbol >= minVocabSymbol and symbol <= maxVocabSymbol and (not super(type(self), self).matches(symbol, minVocabSymbol, maxVocabSymbol))

    def __str__(self):
        return '~' + super(type(self), self).__str__()