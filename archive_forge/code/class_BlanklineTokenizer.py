import re
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.util import regexp_span_tokenize
class BlanklineTokenizer(RegexpTokenizer):
    """
    Tokenize a string, treating any sequence of blank lines as a delimiter.
    Blank lines are defined as lines containing no characters, except for
    space or tab characters.
    """

    def __init__(self):
        RegexpTokenizer.__init__(self, '\\s*\\n\\s*\\n\\s*', gaps=True)