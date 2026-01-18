from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
class ANTLRInputStream(ANTLRStringStream):
    """
    @brief CharStream that reads data from a file-like object.

    This is a char buffer stream that is loaded from a file like object
    all at once when you construct the object.

    All input is consumed from the file, but it is not closed.
    """

    def __init__(self, file, encoding=None):
        """
        @param file A file-like object holding your input. Only the read()
           method must be implemented.

        @param encoding If you set the optional encoding argument, then the
           data will be decoded on the fly.

        """
        if encoding is not None:
            reader = codecs.lookup(encoding)[2]
            file = reader(file)
        data = file.read()
        ANTLRStringStream.__init__(self, data)