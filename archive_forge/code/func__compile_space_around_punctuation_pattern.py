import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
def _compile_space_around_punctuation_pattern(self):
    look_ahead_for_special_token = f'(?=[{self.punctuation_symbols}])'
    look_ahead_to_match_all_except_space = '(?=[^\\s])'
    return re.compile('' + look_ahead_for_special_token + look_ahead_to_match_all_except_space)