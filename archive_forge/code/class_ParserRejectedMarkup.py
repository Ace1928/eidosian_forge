from collections import defaultdict
import itertools
import re
import warnings
import sys
from bs4.element import (
from . import _htmlparser
class ParserRejectedMarkup(Exception):
    """An Exception to be raised when the underlying parser simply
    refuses to parse the given markup.
    """

    def __init__(self, message_or_exception):
        """Explain why the parser rejected the given markup, either
        with a textual explanation or another exception.
        """
        if isinstance(message_or_exception, Exception):
            e = message_or_exception
            message_or_exception = '%s: %s' % (e.__class__.__name__, str(e))
        super(ParserRejectedMarkup, self).__init__(message_or_exception)