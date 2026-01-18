from antlr4.atn.Transition import Transition
class RuleStartState(ATNState):
    __slots__ = ('stopState', 'isPrecedenceRule')

    def __init__(self):
        super().__init__()
        self.stateType = self.RULE_START
        self.stopState = None
        self.isPrecedenceRule = False