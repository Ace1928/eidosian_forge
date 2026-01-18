from antlr4.PredictionContext import PredictionContextCache, SingletonPredictionContext, PredictionContext
from antlr4.InputStream import InputStream
from antlr4.Token import Token
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import LexerATNConfig
from antlr4.atn.ATNSimulator import ATNSimulator
from antlr4.atn.ATNConfigSet import ATNConfigSet, OrderedATNConfigSet
from antlr4.atn.ATNState import RuleStopState, ATNState
from antlr4.atn.LexerActionExecutor import LexerActionExecutor
from antlr4.atn.Transition import Transition
from antlr4.dfa.DFAState import DFAState
from antlr4.error.Errors import LexerNoViableAltException, UnsupportedOperationException
def addDFAEdge(self, from_: DFAState, tk: int, to: DFAState=None, cfgs: ATNConfigSet=None) -> DFAState:
    if to is None and cfgs is not None:
        suppressEdge = cfgs.hasSemanticContext
        cfgs.hasSemanticContext = False
        to = self.addDFAState(cfgs)
        if suppressEdge:
            return to
    if tk < self.MIN_DFA_EDGE or tk > self.MAX_DFA_EDGE:
        return to
    if LexerATNSimulator.debug:
        print('EDGE ' + str(from_) + ' -> ' + str(to) + ' upon ' + chr(tk))
    if from_.edges is None:
        from_.edges = [None] * (self.MAX_DFA_EDGE - self.MIN_DFA_EDGE + 1)
    from_.edges[tk - self.MIN_DFA_EDGE] = to
    return to