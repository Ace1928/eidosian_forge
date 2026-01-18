from antlr4.atn.Transition import Transition
class ATNState(object):
    __slots__ = ('atn', 'stateNumber', 'stateType', 'ruleIndex', 'epsilonOnlyTransitions', 'transitions', 'nextTokenWithinRule')
    INVALID_TYPE = 0
    BASIC = 1
    RULE_START = 2
    BLOCK_START = 3
    PLUS_BLOCK_START = 4
    STAR_BLOCK_START = 5
    TOKEN_START = 6
    RULE_STOP = 7
    BLOCK_END = 8
    STAR_LOOP_BACK = 9
    STAR_LOOP_ENTRY = 10
    PLUS_LOOP_BACK = 11
    LOOP_END = 12
    serializationNames = ['INVALID', 'BASIC', 'RULE_START', 'BLOCK_START', 'PLUS_BLOCK_START', 'STAR_BLOCK_START', 'TOKEN_START', 'RULE_STOP', 'BLOCK_END', 'STAR_LOOP_BACK', 'STAR_LOOP_ENTRY', 'PLUS_LOOP_BACK', 'LOOP_END']
    INVALID_STATE_NUMBER = -1

    def __init__(self):
        self.atn = None
        self.stateNumber = ATNState.INVALID_STATE_NUMBER
        self.stateType = None
        self.ruleIndex = 0
        self.epsilonOnlyTransitions = False
        self.transitions = []
        self.nextTokenWithinRule = None

    def __hash__(self):
        return self.stateNumber

    def __eq__(self, other):
        return isinstance(other, ATNState) and self.stateNumber == other.stateNumber

    def onlyHasEpsilonTransitions(self):
        return self.epsilonOnlyTransitions

    def isNonGreedyExitState(self):
        return False

    def __str__(self):
        return str(self.stateNumber)

    def addTransition(self, trans: Transition, index: int=-1):
        if len(self.transitions) == 0:
            self.epsilonOnlyTransitions = trans.isEpsilon
        elif self.epsilonOnlyTransitions != trans.isEpsilon:
            self.epsilonOnlyTransitions = False
        if index == -1:
            self.transitions.append(trans)
        else:
            self.transitions.insert(index, trans)