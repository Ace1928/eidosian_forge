import html
from typing import List
import regex  # https://github.com/nltk/nltk/issues/2409
from nltk.tokenize.api import TokenizerI
@property
def PHONE_WORD_RE(self) -> 'regex.Pattern':
    """Secondary core TweetTokenizer regex"""
    if not type(self)._PHONE_WORD_RE:
        type(self)._PHONE_WORD_RE = regex.compile(f'({'|'.join(REGEXPS_PHONE)})', regex.VERBOSE | regex.I | regex.UNICODE)
    return type(self)._PHONE_WORD_RE