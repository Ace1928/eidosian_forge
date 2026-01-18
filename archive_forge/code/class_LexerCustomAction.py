from enum import IntEnum
class LexerCustomAction(LexerAction):
    __slots__ = ('ruleIndex', 'actionIndex')

    def __init__(self, ruleIndex: int, actionIndex: int):
        super().__init__(LexerActionType.CUSTOM)
        self.ruleIndex = ruleIndex
        self.actionIndex = actionIndex
        self.isPositionDependent = True

    def execute(self, lexer: Lexer):
        lexer.action(None, self.ruleIndex, self.actionIndex)

    def __hash__(self):
        return hash((self.actionType, self.ruleIndex, self.actionIndex))

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, LexerCustomAction):
            return False
        else:
            return self.ruleIndex == other.ruleIndex and self.actionIndex == other.actionIndex