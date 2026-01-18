from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
class CharStream(IntStream):
    """
    @brief A source of characters for an ANTLR lexer.

    This is an abstract class that must be implemented by a subclass.

    """
    EOF = -1

    def substring(self, start, stop):
        """
        For infinite streams, you don't need this; primarily I'm providing
        a useful interface for action code.  Just make sure actions don't
        use this on streams that don't support it.
        """
        raise NotImplementedError

    def LT(self, i):
        """
        Get the ith character of lookahead.  This is the same usually as
        LA(i).  This will be used for labels in the generated
        lexer code.  I'd prefer to return a char here type-wise, but it's
        probably better to be 32-bit clean and be consistent with LA.
        """
        raise NotImplementedError

    def getLine(self):
        """ANTLR tracks the line information automatically"""
        raise NotImplementedError

    def setLine(self, line):
        """
        Because this stream can rewind, we need to be able to reset the line
        """
        raise NotImplementedError

    def getCharPositionInLine(self):
        """
        The index of the character relative to the beginning of the line 0..n-1
        """
        raise NotImplementedError

    def setCharPositionInLine(self, pos):
        raise NotImplementedError