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
def _markup_resembles_filename(cls, markup):
    """Error-handling method to raise a warning if incoming markup
        resembles a filename.

        :param markup: A bytestring or string.
        :return: Whether or not the markup resembles a filename
            closely enough to justify a warning.
        """
    path_characters = '/\\'
    extensions = ['.html', '.htm', '.xml', '.xhtml', '.txt']
    if isinstance(markup, bytes):
        path_characters = path_characters.encode('utf8')
        extensions = [x.encode('utf8') for x in extensions]
    filelike = False
    if any((x in markup for x in path_characters)):
        filelike = True
    else:
        lower = markup.lower()
        if any((lower.endswith(ext) for ext in extensions)):
            filelike = True
    if filelike:
        warnings.warn('The input looks more like a filename than markup. You may want to open this file and pass the filehandle into Beautiful Soup.', MarkupResemblesLocatorWarning, stacklevel=3)
        return True
    return False