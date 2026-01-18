import sys
class ProxyErrorListener(ErrorListener):

    def __init__(self, delegates):
        super().__init__()
        if delegates is None:
            raise ReferenceError('delegates')
        self.delegates = delegates

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        for delegate in self.delegates:
            delegate.syntaxError(recognizer, offendingSymbol, line, column, msg, e)

    def reportAmbiguity(self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs):
        for delegate in self.delegates:
            delegate.reportAmbiguity(recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs)

    def reportAttemptingFullContext(self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs):
        for delegate in self.delegates:
            delegate.reportAttemptingFullContext(recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs)

    def reportContextSensitivity(self, recognizer, dfa, startIndex, stopIndex, prediction, configs):
        for delegate in self.delegates:
            delegate.reportContextSensitivity(recognizer, dfa, startIndex, stopIndex, prediction, configs)