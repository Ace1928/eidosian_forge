import math
import re
from nltk.tokenize.api import TokenizerI
class TokenTableField:
    """A field in the token table holding parameters for each token,
    used later in the process"""

    def __init__(self, first_pos, ts_occurences, total_count=1, par_count=1, last_par=0, last_tok_seq=None):
        self.__dict__.update(locals())
        del self.__dict__['self']