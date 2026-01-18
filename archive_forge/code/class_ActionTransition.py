from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.SemanticContext import Predicate, PrecedencePredicate
from antlr4.atn.ATNState import *
class ActionTransition(Transition):
    __slots__ = ('serializationType', 'ruleIndex', 'actionIndex', 'isCtxDependent')

    def __init__(self, target: ATNState, ruleIndex: int, actionIndex: int=-1, isCtxDependent: bool=False):
        super().__init__(target)
        self.serializationType = self.ACTION
        self.ruleIndex = ruleIndex
        self.actionIndex = actionIndex
        self.isCtxDependent = isCtxDependent
        self.isEpsilon = True

    def matches(self, symbol: int, minVocabSymbol: int, maxVocabSymbol: int):
        return False

    def __str__(self):
        return 'action_' + self.ruleIndex + ':' + self.actionIndex