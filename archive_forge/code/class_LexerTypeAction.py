from enum import IntEnum
class LexerTypeAction(LexerAction):
    __slots__ = 'type'

    def __init__(self, type: int):
        super().__init__(LexerActionType.TYPE)
        self.type = type

    def execute(self, lexer: Lexer):
        lexer.type = self.type

    def __hash__(self):
        return hash((self.actionType, self.type))

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, LexerTypeAction):
            return False
        else:
            return self.type == other.type

    def __str__(self):
        return 'type(' + str(self.type) + ')'