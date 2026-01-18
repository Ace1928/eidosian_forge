from enum import IntEnum
class LexerAction(object):
    __slots__ = ('actionType', 'isPositionDependent')

    def __init__(self, action: LexerActionType):
        self.actionType = action
        self.isPositionDependent = False

    def __hash__(self):
        return hash(self.actionType)

    def __eq__(self, other):
        return self is other