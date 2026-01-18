import re
import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.xmldocs import *
from nltk.internals import ElementWrapper
from nltk.tag import map_tag
from nltk.util import LazyConcatenation
@staticmethod
def _simplify_username(word):
    if 'User' in word:
        word = 'U' + word.split('User', 1)[1]
    elif isinstance(word, bytes):
        word = word.decode('ascii')
    return word