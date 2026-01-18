import re
import sys
from collections import Counter, defaultdict, namedtuple
from functools import reduce
from math import log
from nltk.collocations import BigramCollocationFinder
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.metrics import BigramAssocMeasures, f_measure
from nltk.probability import ConditionalFreqDist as CFD
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.util import LazyConcatenation, tokenwrap
class TokenSearcher:
    """
    A class that makes it easier to use regular expressions to search
    over tokenized strings.  The tokenized string is converted to a
    string where tokens are marked with angle brackets -- e.g.,
    ``'<the><window><is><still><open>'``.  The regular expression
    passed to the ``findall()`` method is modified to treat angle
    brackets as non-capturing parentheses, in addition to matching the
    token boundaries; and to have ``'.'`` not match the angle brackets.
    """

    def __init__(self, tokens):
        self._raw = ''.join(('<' + w + '>' for w in tokens))

    def findall(self, regexp):
        """
        Find instances of the regular expression in the text.
        The text is a list of tokens, and a regexp pattern to match
        a single token must be surrounded by angle brackets.  E.g.

        >>> from nltk.text import TokenSearcher
        >>> from nltk.book import text1, text5, text9
        >>> text5.findall("<.*><.*><bro>")
        you rule bro; telling you bro; u twizted bro
        >>> text1.findall("<a>(<.*>)<man>")
        monied; nervous; dangerous; white; white; white; pious; queer; good;
        mature; white; Cape; great; wise; wise; butterless; white; fiendish;
        pale; furious; better; certain; complete; dismasted; younger; brave;
        brave; brave; brave
        >>> text9.findall("<th.*>{3,}")
        thread through those; the thought that; that the thing; the thing
        that; that that thing; through these than through; them that the;
        through the thick; them that they; thought that the

        :param regexp: A regular expression
        :type regexp: str
        """
        regexp = re.sub('\\s', '', regexp)
        regexp = re.sub('<', '(?:<(?:', regexp)
        regexp = re.sub('>', ')>)', regexp)
        regexp = re.sub('(?<!\\\\)\\.', '[^>]', regexp)
        hits = re.findall(regexp, self._raw)
        for h in hits:
            if not h.startswith('<') and h.endswith('>'):
                raise ValueError('Bad regexp for TokenSearcher.findall')
        hits = [h[1:-1].split('><') for h in hits]
        return hits