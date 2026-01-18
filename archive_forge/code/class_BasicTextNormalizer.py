import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union
import regex
class BasicTextNormalizer:

    def __init__(self, remove_diacritics: bool=False, split_letters: bool=False):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub('[<\\[][^>\\]]*[>\\]]', '', s)
        s = re.sub('\\(([^)]+?)\\)', '', s)
        s = self.clean(s).lower()
        if self.split_letters:
            s = ' '.join(regex.findall('\\X', s, regex.U))
        s = re.sub('\\s+', ' ', s)
        return s