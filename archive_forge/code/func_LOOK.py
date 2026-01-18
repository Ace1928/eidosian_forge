from antlr4.IntervalSet import IntervalSet
from antlr4.Token import Token
from antlr4.PredictionContext import PredictionContext, SingletonPredictionContext, PredictionContextFromRuleContext
from antlr4.RuleContext import RuleContext
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNState import ATNState, RuleStopState
from antlr4.atn.Transition import WildcardTransition, NotSetTransition, AbstractPredicateTransition, RuleTransition
def LOOK(self, s: ATNState, stopState: ATNState=None, ctx: RuleContext=None):
    r = IntervalSet()
    seeThruPreds = True
    lookContext = PredictionContextFromRuleContext(s.atn, ctx) if ctx is not None else None
    self._LOOK(s, stopState, lookContext, r, set(), set(), seeThruPreds, True)
    return r