from typing import List, Tuple
import pytest
from nltk.tokenize import (
class TestPunktTokenizeWordsMock:

    def word_tokenize(self, s):
        return iter([])