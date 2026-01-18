from typing import List, Tuple
import pytest
from nltk.tokenize import (
class BengaliLanguageVars(punkt.PunktLanguageVars):
    sent_end_chars = ('.', '?', '!', '।')