from io import StringIO
from antlr4.PredictionContext import PredictionContext
from antlr4.atn.ATNState import ATNState, DecisionState
from antlr4.atn.LexerActionExecutor import LexerActionExecutor
from antlr4.atn.SemanticContext import SemanticContext
class LexerATNConfig(ATNConfig):
    __slots__ = ('lexerActionExecutor', 'passedThroughNonGreedyDecision')

    def __init__(self, state: ATNState, alt: int=None, context: PredictionContext=None, semantic: SemanticContext=SemanticContext.NONE, lexerActionExecutor: LexerActionExecutor=None, config: LexerATNConfig=None):
        super().__init__(state=state, alt=alt, context=context, semantic=semantic, config=config)
        if config is not None:
            if lexerActionExecutor is None:
                lexerActionExecutor = config.lexerActionExecutor
        self.lexerActionExecutor = lexerActionExecutor
        self.passedThroughNonGreedyDecision = False if config is None else self.checkNonGreedyDecision(config, state)

    def __hash__(self):
        return hash((self.state.stateNumber, self.alt, self.context, self.semanticContext, self.passedThroughNonGreedyDecision, self.lexerActionExecutor))

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, LexerATNConfig):
            return False
        if self.passedThroughNonGreedyDecision != other.passedThroughNonGreedyDecision:
            return False
        if not self.lexerActionExecutor == other.lexerActionExecutor:
            return False
        return super().__eq__(other)

    def hashCodeForConfigSet(self):
        return hash(self)

    def equalsForConfigSet(self, other):
        return self == other

    def checkNonGreedyDecision(self, source: LexerATNConfig, target: ATNState):
        return source.passedThroughNonGreedyDecision or (isinstance(target, DecisionState) and target.nonGreedy)