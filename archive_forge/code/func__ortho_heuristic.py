import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _ortho_heuristic(self, aug_tok: PunktToken) -> Union[bool, str]:
    """
        Decide whether the given token is the first token in a sentence.
        """
    if aug_tok.tok in self.PUNCTUATION:
        return False
    ortho_context = self._params.ortho_context[aug_tok.type_no_sentperiod]
    if aug_tok.first_upper and ortho_context & _ORTHO_LC and (not ortho_context & _ORTHO_MID_UC):
        return True
    if aug_tok.first_lower and (ortho_context & _ORTHO_UC or not ortho_context & _ORTHO_BEG_LC):
        return False
    return 'unknown'