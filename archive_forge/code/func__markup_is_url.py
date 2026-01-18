from collections import Counter
import os
import re
import sys
import traceback
import warnings
from .builder import (
from .dammit import UnicodeDammit
from .element import (
@classmethod
def _markup_is_url(cls, markup):
    """Error-handling method to raise a warning if incoming markup looks
        like a URL.

        :param markup: A string.
        :return: Whether or not the markup resembles a URL
            closely enough to justify a warning.
        """
    if isinstance(markup, bytes):
        space = b' '
        cant_start_with = (b'http:', b'https:')
    elif isinstance(markup, str):
        space = ' '
        cant_start_with = ('http:', 'https:')
    else:
        return False
    if any((markup.startswith(prefix) for prefix in cant_start_with)):
        if not space in markup:
            warnings.warn('The input looks more like a URL than markup. You may want to use an HTTP client like requests to get the document behind the URL, and feed that document to Beautiful Soup.', MarkupResemblesLocatorWarning, stacklevel=3)
            return True
    return False