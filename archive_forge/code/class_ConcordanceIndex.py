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
class ConcordanceIndex:
    """
    An index that can be used to look up the offset locations at which
    a given word occurs in a document.
    """

    def __init__(self, tokens, key=lambda x: x):
        """
        Construct a new concordance index.

        :param tokens: The document (list of tokens) that this
            concordance index was created from.  This list can be used
            to access the context of a given word occurrence.
        :param key: A function that maps each token to a normalized
            version that will be used as a key in the index.  E.g., if
            you use ``key=lambda s:s.lower()``, then the index will be
            case-insensitive.
        """
        self._tokens = tokens
        'The document (list of tokens) that this concordance index\n           was created from.'
        self._key = key
        'Function mapping each token to an index key (or None).'
        self._offsets = defaultdict(list)
        'Dictionary mapping words (or keys) to lists of offset indices.'
        for index, word in enumerate(tokens):
            word = self._key(word)
            self._offsets[word].append(index)

    def tokens(self):
        """
        :rtype: list(str)
        :return: The document that this concordance index was
            created from.
        """
        return self._tokens

    def offsets(self, word):
        """
        :rtype: list(int)
        :return: A list of the offset positions at which the given
            word occurs.  If a key function was specified for the
            index, then given word's key will be looked up.
        """
        word = self._key(word)
        return self._offsets[word]

    def __repr__(self):
        return '<ConcordanceIndex for %d tokens (%d types)>' % (len(self._tokens), len(self._offsets))

    def find_concordance(self, word, width=80):
        """
        Find all concordance lines given the query word.

        Provided with a list of words, these will be found as a phrase.
        """
        if isinstance(word, list):
            phrase = word
        else:
            phrase = [word]
        half_width = (width - len(' '.join(phrase)) - 2) // 2
        context = width // 4
        concordance_list = []
        offsets = self.offsets(phrase[0])
        for i, word in enumerate(phrase[1:]):
            word_offsets = {offset - i - 1 for offset in self.offsets(word)}
            offsets = sorted(word_offsets.intersection(offsets))
        if offsets:
            for i in offsets:
                query_word = ' '.join(self._tokens[i:i + len(phrase)])
                left_context = self._tokens[max(0, i - context):i]
                right_context = self._tokens[i + len(phrase):i + context]
                left_print = ' '.join(left_context)[-half_width:]
                right_print = ' '.join(right_context)[:half_width]
                line_print = ' '.join([left_print, query_word, right_print])
                concordance_line = ConcordanceLine(left_context, query_word, right_context, i, left_print, right_print, line_print)
                concordance_list.append(concordance_line)
        return concordance_list

    def print_concordance(self, word, width=80, lines=25):
        """
        Print concordance lines given the query word.
        :param word: The target word or phrase (a list of strings)
        :type word: str or list
        :param lines: The number of lines to display (default=25)
        :type lines: int
        :param width: The width of each line, in characters (default=80)
        :type width: int
        :param save: The option to save the concordance.
        :type save: bool
        """
        concordance_list = self.find_concordance(word, width=width)
        if not concordance_list:
            print('no matches')
        else:
            lines = min(lines, len(concordance_list))
            print(f'Displaying {lines} of {len(concordance_list)} matches:')
            for i, concordance_line in enumerate(concordance_list[:lines]):
                print(concordance_line.line)