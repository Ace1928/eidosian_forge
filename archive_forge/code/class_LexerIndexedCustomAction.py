from enum import IntEnum
class LexerIndexedCustomAction(LexerAction):
    __slots__ = ('offset', 'action')

    def __init__(self, offset: int, action: LexerAction):
        super().__init__(action.actionType)
        self.offset = offset
        self.action = action
        self.isPositionDependent = True

    def execute(self, lexer: Lexer):
        self.action.execute(lexer)

    def __hash__(self):
        return hash((self.actionType, self.offset, self.action))

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, LexerIndexedCustomAction):
            return False
        else:
            return self.offset == other.offset and self.action == other.action