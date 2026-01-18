import itertools
import os
import re
from string import Template
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from tokenizers import Encoding, Tokenizer
class CharStateKey(NamedTuple):
    token_ix: Optional[int]
    anno_ix: Optional[int]