from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
class ReplaceOp(RewriteOperation):
    """
    @brief Internal helper class.

    I'm going to try replacing range from x..y with (y-x)+1 ReplaceOp
    instructions.
    """

    def __init__(self, stream, first, last, text):
        RewriteOperation.__init__(self, stream, first, text)
        self.lastIndex = last

    def execute(self, buf):
        if self.text is not None:
            buf.write(self.text)
        return self.lastIndex + 1

    def toString(self):
        return '<ReplaceOp@%d..%d:"%s">' % (self.index, self.lastIndex, self.text)
    __str__ = toString
    __repr__ = toString