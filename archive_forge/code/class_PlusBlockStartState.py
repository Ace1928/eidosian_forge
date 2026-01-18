from antlr4.atn.Transition import Transition
class PlusBlockStartState(BlockStartState):
    __slots__ = 'loopBackState'

    def __init__(self):
        super().__init__()
        self.stateType = self.PLUS_BLOCK_START
        self.loopBackState = None