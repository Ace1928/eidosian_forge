import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for the Python language.
    """
    keywords = ['and', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'exec', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'not', 'or', 'pass', 'print', 'raise', 'return', 'try', 'while', 'yield', 'None', 'True', 'False', 'async', 'await']
    operators = ['=', '==', '!=', '<', '<=', '>', '>=', '\\+', '-', '\\*', '/', '//', '\\%', '\\*\\*', '\\+=', '-=', '\\*=', '/=', '\\%=', '\\^', '\\|', '\\&', '\\~', '>>', '<<']
    braces = ['\\{', '\\}', '\\(', '\\)', '\\[', '\\]']

    def __init__(self, document):
        QSyntaxHighlighter.__init__(self, document)
        self.tri_single = (QRegExp("'''"), 1, 'string2')
        self.tri_double = (QRegExp('"""'), 2, 'string2')
        rules = []
        rules += [('\\b%s\\b' % w, 0, 'keyword') for w in PythonHighlighter.keywords]
        rules += [('%s' % o, 0, 'operator') for o in PythonHighlighter.operators]
        rules += [('%s' % b, 0, 'brace') for b in PythonHighlighter.braces]
        rules += [('\\bself\\b', 0, 'self'), ('\\bdef\\b\\s*(\\w+)', 1, 'defclass'), ('\\bclass\\b\\s*(\\w+)', 1, 'defclass'), ('\\b[+-]?[0-9]+[lL]?\\b', 0, 'numbers'), ('\\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\\b', 0, 'numbers'), ('\\b[+-]?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\\b', 0, 'numbers'), ('"[^"\\\\]*(\\\\.[^"\\\\]*)*"', 0, 'string'), ("'[^'\\\\]*(\\\\.[^'\\\\]*)*'", 0, 'string'), ('#[^\\n]*', 0, 'comment')]
        self.rules = [(QRegExp(pat), index, fmt) for pat, index, fmt in rules]

    @property
    def styles(self):
        app = QtWidgets.QApplication.instance()
        return DARK_STYLES if app.property('darkMode') else LIGHT_STYLES

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text.
        """
        for expression, nth, format in self.rules:
            index = expression.indexIn(text, 0)
            format = self.styles[format]
            while index >= 0:
                index = expression.pos(nth)
                length = len(expression.cap(nth))
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)
        self.setCurrentBlockState(0)
        in_multiline = self.match_multiline(text, *self.tri_single)
        if not in_multiline:
            in_multiline = self.match_multiline(text, *self.tri_double)

    def match_multiline(self, text, delimiter, in_state, style):
        """Do highlighting of multi-line strings. ``delimiter`` should be a
        ``QRegExp`` for triple-single-quotes or triple-double-quotes, and
        ``in_state`` should be a unique integer to represent the corresponding
        state changes when inside those strings. Returns True if we're still
        inside a multi-line string when this function is finished.
        """
        if self.previousBlockState() == in_state:
            start = 0
            add = 0
        else:
            start = delimiter.indexIn(text)
            add = delimiter.matchedLength()
        while start >= 0:
            end = delimiter.indexIn(text, start + add)
            if end >= add:
                length = end - start + add + delimiter.matchedLength()
                self.setCurrentBlockState(0)
            else:
                self.setCurrentBlockState(in_state)
                length = len(text) - start + add
            self.setFormat(start, length, self.styles[style])
            start = delimiter.indexIn(text, start + length)
        if self.currentBlockState() == in_state:
            return True
        else:
            return False