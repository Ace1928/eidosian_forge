from typing import List, Tuple
import pytest
from nltk.tokenize import (
class ExtLangVars(punkt.PunktLanguageVars):
    sent_end_chars = ('.', '?', '!', '^')