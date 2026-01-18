from enum import IntEnum
class LexerPopModeAction(LexerAction):
    INSTANCE = None

    def __init__(self):
        super().__init__(LexerActionType.POP_MODE)

    def execute(self, lexer: Lexer):
        lexer.popMode()

    def __str__(self):
        return 'popMode'