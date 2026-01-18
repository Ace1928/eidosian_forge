from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
class ANTLRFileStream(ANTLRStringStream):
    """
    @brief CharStream that opens a file to read the data.

    This is a char buffer stream that is loaded from a file
    all at once when you construct the object.
    """

    def __init__(self, fileName, encoding=None):
        """
        @param fileName The path to the file to be opened. The file will be
           opened with mode 'rb'.

        @param encoding If you set the optional encoding argument, then the
           data will be decoded on the fly.

        """
        self.fileName = fileName
        fp = codecs.open(fileName, 'rb', encoding)
        try:
            data = fp.read()
        finally:
            fp.close()
        ANTLRStringStream.__init__(self, data)

    def getSourceName(self):
        """Deprecated, access o.fileName directly."""
        return self.fileName