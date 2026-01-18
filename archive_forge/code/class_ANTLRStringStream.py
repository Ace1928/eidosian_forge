from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
class ANTLRStringStream(CharStream):
    """
    @brief CharStream that pull data from a unicode string.

    A pretty quick CharStream that pulls all data from an array
    directly.  Every method call counts in the lexer.

    """

    def __init__(self, data):
        """
        @param data This should be a unicode string holding the data you want
           to parse. If you pass in a byte string, the Lexer will choke on
           non-ascii data.

        """
        CharStream.__init__(self)
        self.strdata = six.text_type(data)
        self.data = [ord(c) for c in self.strdata]
        self.n = len(data)
        self.p = 0
        self.line = 1
        self.charPositionInLine = 0
        self._markers = []
        self.lastMarker = None
        self.markDepth = 0
        self.name = None

    def reset(self):
        """
        Reset the stream so that it's in the same state it was
        when the object was created *except* the data array is not
        touched.
        """
        self.p = 0
        self.line = 1
        self.charPositionInLine = 0
        self._markers = []

    def consume(self):
        try:
            if self.data[self.p] == 10:
                self.line += 1
                self.charPositionInLine = 0
            else:
                self.charPositionInLine += 1
            self.p += 1
        except IndexError:
            pass

    def LA(self, i):
        if i == 0:
            return 0
        if i < 0:
            i += 1
        try:
            return self.data[self.p + i - 1]
        except IndexError:
            return EOF

    def LT(self, i):
        if i == 0:
            return 0
        if i < 0:
            i += 1
        try:
            return self.strdata[self.p + i - 1]
        except IndexError:
            return EOF

    def index(self):
        """
        Return the current input symbol index 0..n where n indicates the
        last symbol has been read.  The index is the index of char to
        be returned from LA(1).
        """
        return self.p

    def size(self):
        return self.n

    def mark(self):
        state = (self.p, self.line, self.charPositionInLine)
        try:
            self._markers[self.markDepth] = state
        except IndexError:
            self._markers.append(state)
        self.markDepth += 1
        self.lastMarker = self.markDepth
        return self.lastMarker

    def rewind(self, marker=None):
        if marker is None:
            marker = self.lastMarker
        p, line, charPositionInLine = self._markers[marker - 1]
        self.seek(p)
        self.line = line
        self.charPositionInLine = charPositionInLine
        self.release(marker)

    def release(self, marker=None):
        if marker is None:
            marker = self.lastMarker
        self.markDepth = marker - 1

    def seek(self, index):
        """
        consume() ahead until p==index; can't just set p=index as we must
        update line and charPositionInLine.
        """
        if index <= self.p:
            self.p = index
            return
        while self.p < index:
            self.consume()

    def substring(self, start, stop):
        return self.strdata[start:stop + 1]

    def getLine(self):
        """Using setter/getter methods is deprecated. Use o.line instead."""
        return self.line

    def getCharPositionInLine(self):
        """
        Using setter/getter methods is deprecated. Use o.charPositionInLine
        instead.
        """
        return self.charPositionInLine

    def setLine(self, line):
        """Using setter/getter methods is deprecated. Use o.line instead."""
        self.line = line

    def setCharPositionInLine(self, pos):
        """
        Using setter/getter methods is deprecated. Use o.charPositionInLine
        instead.
        """
        self.charPositionInLine = pos

    def getSourceName(self):
        return self.name