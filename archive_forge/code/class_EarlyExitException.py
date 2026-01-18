from antlr3.constants import INVALID_TOKEN_TYPE
class EarlyExitException(RecognitionException):
    """@brief The recognizer did not match anything for a (..)+ loop."""

    def __init__(self, decisionNumber, input):
        RecognitionException.__init__(self, input)
        self.decisionNumber = decisionNumber