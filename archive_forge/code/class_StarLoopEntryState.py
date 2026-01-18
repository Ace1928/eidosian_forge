from antlr4.atn.Transition import Transition
class StarLoopEntryState(DecisionState):
    __slots__ = ('loopBackState', 'isPrecedenceDecision')

    def __init__(self):
        super().__init__()
        self.stateType = self.STAR_LOOP_ENTRY
        self.loopBackState = None
        self.isPrecedenceDecision = None