from antlr4.atn.Transition import Transition
class RuleStopState(ATNState):

    def __init__(self):
        super().__init__()
        self.stateType = self.RULE_STOP