import html
from typing import List
import regex  # https://github.com/nltk/nltk/issues/2409
from nltk.tokenize.api import TokenizerI
@property
def WORD_RE(self) -> 'regex.Pattern':
    """Core TweetTokenizer regex"""
    if not type(self)._WORD_RE:
        type(self)._WORD_RE = regex.compile(f'({'|'.join(REGEXPS)})', regex.VERBOSE | regex.I | regex.UNICODE)
    return type(self)._WORD_RE