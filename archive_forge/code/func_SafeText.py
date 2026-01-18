from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import locale
import os
import sys
import unicodedata
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import encoding as encoding_util
import six
def SafeText(data, encoding=None, escape=True):
    b"Converts the data to a text string compatible with the given encoding.\n\n  This works the same way as Decode() below except it guarantees that any\n  characters in the resulting text string can be re-encoded using the given\n  encoding (or GetConsoleAttr().GetEncoding() if None is given). This means\n  that the string will be safe to print to sys.stdout (for example) without\n  getting codec exceptions if the user's terminal doesn't support the encoding\n  used by the source of the text.\n\n  Args:\n    data: Any bytes, string, or object that has str() or unicode() methods.\n    encoding: The encoding name to ensure compatibility with. Defaults to\n      GetConsoleAttr().GetEncoding().\n    escape: Replace unencodable characters with a \\uXXXX or \\xXX equivalent if\n      True. Otherwise replace unencodable characters with an appropriate unknown\n      character, '?' for ASCII, and the unicode unknown replacement character\n      \\uFFFE for unicode.\n\n  Returns:\n    A text string representation of the data, but modified to remove any\n    characters that would result in an encoding exception with the target\n    encoding. In the worst case, with escape=False, it will contain only ?\n    characters.\n  "
    if data is None:
        return 'None'
    encoding = encoding or GetConsoleAttr().GetEncoding()
    string = encoding_util.Decode(data, encoding=encoding)
    try:
        string.encode(encoding)
        return string
    except UnicodeError:
        return string.encode(encoding, 'backslashreplace' if escape else 'replace').decode(encoding)