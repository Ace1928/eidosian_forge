import sys
from antlr4 import DFA
from antlr4.PredictionContext import PredictionContextCache, PredictionContext, SingletonPredictionContext, \
from antlr4.BufferedTokenStream import TokenStream
from antlr4.Parser import Parser
from antlr4.ParserRuleContext import ParserRuleContext
from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.Utils import str_list
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.atn.ATNSimulator import ATNSimulator
from antlr4.atn.ATNState import StarLoopEntryState, DecisionState, RuleStopState, ATNState
from antlr4.atn.PredictionMode import PredictionMode
from antlr4.atn.SemanticContext import SemanticContext, AND, andContext, orContext
from antlr4.atn.Transition import Transition, RuleTransition, ActionTransition, PrecedencePredicateTransition, \
from antlr4.dfa.DFAState import DFAState, PredPrediction
from antlr4.error.Errors import NoViableAltException
def adaptivePredict(self, input: TokenStream, decision: int, outerContext: ParserRuleContext):
    if ParserATNSimulator.debug or ParserATNSimulator.debug_list_atn_decisions:
        print('adaptivePredict decision ' + str(decision) + ' exec LA(1)==' + self.getLookaheadName(input) + ' line ' + str(input.LT(1).line) + ':' + str(input.LT(1).column))
    self._input = input
    self._startIndex = input.index
    self._outerContext = outerContext
    dfa = self.decisionToDFA[decision]
    self._dfa = dfa
    m = input.mark()
    index = input.index
    try:
        if dfa.precedenceDfa:
            s0 = dfa.getPrecedenceStartState(self.parser.getPrecedence())
        else:
            s0 = dfa.s0
        if s0 is None:
            if outerContext is None:
                outerContext = ParserRuleContext.EMPTY
            if ParserATNSimulator.debug or ParserATNSimulator.debug_list_atn_decisions:
                print('predictATN decision ' + str(dfa.decision) + ' exec LA(1)==' + self.getLookaheadName(input) + ', outerContext=' + outerContext.toString(self.parser.literalNames, None))
            fullCtx = False
            s0_closure = self.computeStartState(dfa.atnStartState, ParserRuleContext.EMPTY, fullCtx)
            if dfa.precedenceDfa:
                dfa.s0.configs = s0_closure
                s0_closure = self.applyPrecedenceFilter(s0_closure)
                s0 = self.addDFAState(dfa, DFAState(configs=s0_closure))
                dfa.setPrecedenceStartState(self.parser.getPrecedence(), s0)
            else:
                s0 = self.addDFAState(dfa, DFAState(configs=s0_closure))
                dfa.s0 = s0
        alt = self.execATN(dfa, s0, input, index, outerContext)
        if ParserATNSimulator.debug:
            print('DFA after predictATN: ' + dfa.toString(self.parser.literalNames))
        return alt
    finally:
        self._dfa = None
        self.mergeCache = None
        input.seek(index)
        input.release(m)