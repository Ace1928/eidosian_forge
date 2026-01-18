import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
class PunktLanguageVars:
    """
    Stores variables, mostly regular expressions, which may be
    language-dependent for correct application of the algorithm.
    An extension of this class may modify its properties to suit
    a language other than English; an instance can then be passed
    as an argument to PunktSentenceTokenizer and PunktTrainer
    constructors.
    """
    __slots__ = ('_re_period_context', '_re_word_tokenizer')

    def __getstate__(self):
        return 1

    def __setstate__(self, state):
        return 1
    sent_end_chars = ('.', '?', '!')
    'Characters which are candidates for sentence boundaries'

    @property
    def _re_sent_end_chars(self):
        return '[%s]' % re.escape(''.join(self.sent_end_chars))
    internal_punctuation = ',:;'
    'sentence internal punctuation, which indicates an abbreviation if\n    preceded by a period-final token.'
    re_boundary_realignment = re.compile('["\\\')\\]}]+?(?:\\s+|(?=--)|$)', re.MULTILINE)
    'Used to realign punctuation that should be included in a sentence\n    although it follows the period (or ?, !).'
    _re_word_start = '[^\\(\\"\\`{\\[:;&\\#\\*@\\)}\\]\\-,]'
    'Excludes some characters from starting word tokens'

    @property
    def _re_non_word_chars(self):
        return '(?:[)\\";}\\]\\*:@\\\'\\({\\[%s])' % re.escape(''.join(set(self.sent_end_chars) - {'.'}))
    'Characters that cannot appear within words'
    _re_multi_char_punct = '(?:\\-{2,}|\\.{2,}|(?:\\.\\s){2,}\\.)'
    'Hyphen and ellipsis are multi-character punctuation'
    _word_tokenize_fmt = "(\n        %(MultiChar)s\n        |\n        (?=%(WordStart)s)\\S+?  # Accept word characters until end is found\n        (?= # Sequences marking a word's end\n            \\s|                                 # White-space\n            $|                                  # End-of-string\n            %(NonWord)s|%(MultiChar)s|          # Punctuation\n            ,(?=$|\\s|%(NonWord)s|%(MultiChar)s) # Comma if at end of word\n        )\n        |\n        \\S\n    )"
    'Format of a regular expression to split punctuation from words,\n    excluding period.'

    def _word_tokenizer_re(self):
        """Compiles and returns a regular expression for word tokenization"""
        try:
            return self._re_word_tokenizer
        except AttributeError:
            self._re_word_tokenizer = re.compile(self._word_tokenize_fmt % {'NonWord': self._re_non_word_chars, 'MultiChar': self._re_multi_char_punct, 'WordStart': self._re_word_start}, re.UNICODE | re.VERBOSE)
            return self._re_word_tokenizer

    def word_tokenize(self, s):
        """Tokenize a string to split off punctuation other than periods"""
        return self._word_tokenizer_re().findall(s)
    _period_context_fmt = '\n        %(SentEndChars)s             # a potential sentence ending\n        (?=(?P<after_tok>\n            %(NonWord)s              # either other punctuation\n            |\n            \\s+(?P<next_tok>\\S+)     # or whitespace and some other token\n        ))'
    'Format of a regular expression to find contexts including possible\n    sentence boundaries. Matches token which the possible sentence boundary\n    ends, and matches the following token within a lookahead expression.'

    def period_context_re(self):
        """Compiles and returns a regular expression to find contexts
        including possible sentence boundaries."""
        try:
            return self._re_period_context
        except:
            self._re_period_context = re.compile(self._period_context_fmt % {'NonWord': self._re_non_word_chars, 'SentEndChars': self._re_sent_end_chars}, re.UNICODE | re.VERBOSE)
            return self._re_period_context