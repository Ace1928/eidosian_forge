from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.SemanticContext import Predicate, PrecedencePredicate
from antlr4.atn.ATNState import *
class AtomTransition(Transition):
    __slots__ = ('label_', 'serializationType')

    def __init__(self, target: ATNState, label: int):
        super().__init__(target)
        self.label_ = label
        self.label = self.makeLabel()
        self.serializationType = self.ATOM

    def makeLabel(self):
        s = IntervalSet()
        s.addOne(self.label_)
        return s

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        return self.label_ == symbol

    def __str__(self):
        return str(self.label_)