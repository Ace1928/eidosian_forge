import sys
class ConsoleErrorListener(ErrorListener):
    INSTANCE = None

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        print('line ' + str(line) + ':' + str(column) + ' ' + msg, file=sys.stderr)