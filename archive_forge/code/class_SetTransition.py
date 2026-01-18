from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.SemanticContext import Predicate, PrecedencePredicate
from antlr4.atn.ATNState import *
class SetTransition(Transition):
    __slots__ = 'serializationType'

    def __init__(self, target: ATNState, set: IntervalSet):
        super().__init__(target)
        self.serializationType = self.SET
        if set is not None:
            self.label = set
        else:
            self.label = IntervalSet()
            self.label.addRange(range(Token.INVALID_TYPE, Token.INVALID_TYPE + 1))

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        return symbol in self.label

    def __str__(self):
        return str(self.label)