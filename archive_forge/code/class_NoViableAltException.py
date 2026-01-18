from antlr3.constants import INVALID_TOKEN_TYPE
class NoViableAltException(RecognitionException):
    """@brief Unable to decide which alternative to choose."""

    def __init__(self, grammarDecisionDescription, decisionNumber, stateNumber, input):
        RecognitionException.__init__(self, input)
        self.grammarDecisionDescription = grammarDecisionDescription
        self.decisionNumber = decisionNumber
        self.stateNumber = stateNumber

    def __str__(self):
        return 'NoViableAltException(%r!=[%r])' % (self.unexpectedType, self.grammarDecisionDescription)
    __repr__ = __str__