from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.SemanticContext import Predicate, PrecedencePredicate
from antlr4.atn.ATNState import *
class RangeTransition(Transition):
    __slots__ = ('serializationType', 'start', 'stop')

    def __init__(self, target: ATNState, start: int, stop: int):
        super().__init__(target)
        self.serializationType = self.RANGE
        self.start = start
        self.stop = stop
        self.label = self.makeLabel()

    def makeLabel(self):
        s = IntervalSet()
        s.addRange(range(self.start, self.stop + 1))
        return s

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        return symbol >= self.start and symbol <= self.stop

    def __str__(self):
        return "'" + chr(self.start) + "'..'" + chr(self.stop) + "'"