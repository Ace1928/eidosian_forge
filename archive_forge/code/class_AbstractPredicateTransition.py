from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.atn.SemanticContext import Predicate, PrecedencePredicate
from antlr4.atn.ATNState import *
class AbstractPredicateTransition(Transition):

    def __init__(self, target: ATNState):
        super().__init__(target)