from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
class BadDirectiveError(TemplateSyntaxError):
    """Exception raised when an unknown directive is encountered when parsing
    a template.
    
    An unknown directive is any attribute using the namespace for directives,
    with a local name that doesn't match any registered directive.
    """

    def __init__(self, name, filename=None, lineno=-1):
        """Create the exception
        
        :param name: the name of the directive
        :param filename: the filename of the template
        :param lineno: the number of line in the template at which the error
                       occurred
        """
        TemplateSyntaxError.__init__(self, 'bad directive "%s"' % name, filename, lineno)