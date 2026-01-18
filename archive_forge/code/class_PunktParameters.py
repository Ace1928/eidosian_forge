import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
class PunktParameters:
    """Stores data used to perform sentence boundary detection with Punkt."""

    def __init__(self):
        self.abbrev_types = set()
        'A set of word types for known abbreviations.'
        self.collocations = set()
        "A set of word type tuples for known common collocations\n        where the first word ends in a period.  E.g., ('S.', 'Bach')\n        is a common collocation in a text that discusses 'Johann\n        S. Bach'.  These count as negative evidence for sentence\n        boundaries."
        self.sent_starters = set()
        'A set of word types for words that often appear at the\n        beginning of sentences.'
        self.ortho_context = defaultdict(int)
        'A dictionary mapping word types to the set of orthographic\n        contexts that word type appears in.  Contexts are represented\n        by adding orthographic context flags: ...'

    def clear_abbrevs(self):
        self.abbrev_types = set()

    def clear_collocations(self):
        self.collocations = set()

    def clear_sent_starters(self):
        self.sent_starters = set()

    def clear_ortho_context(self):
        self.ortho_context = defaultdict(int)

    def add_ortho_context(self, typ, flag):
        self.ortho_context[typ] |= flag

    def _debug_ortho_context(self, typ):
        context = self.ortho_context[typ]
        if context & _ORTHO_BEG_UC:
            yield 'BEG-UC'
        if context & _ORTHO_MID_UC:
            yield 'MID-UC'
        if context & _ORTHO_UNK_UC:
            yield 'UNK-UC'
        if context & _ORTHO_BEG_LC:
            yield 'BEG-LC'
        if context & _ORTHO_MID_LC:
            yield 'MID-LC'
        if context & _ORTHO_UNK_LC:
            yield 'UNK-LC'