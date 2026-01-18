import re
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import srsly
from thinc.api import Model
from ... import util
from ...errors import Errors
from ...language import BaseDefaults, Language
from ...pipeline import Morphologizer
from ...pipeline.morphologizer import DEFAULT_MORPH_MODEL
from ...scorer import Scorer
from ...symbols import POS
from ...tokens import Doc, MorphAnalysis
from ...training import validate_examples
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .stop_words import STOP_WORDS
from .syntax_iterators import SYNTAX_ITERATORS
from .tag_bigram_map import TAG_BIGRAM_MAP
from .tag_map import TAG_MAP
from .tag_orth_map import TAG_ORTH_MAP
def _get_sub_tokens(self, sudachipy_tokens):
    if not self.need_subtokens:
        return None
    sub_tokens_list = []
    for token in sudachipy_tokens:
        sub_a = token.split(self.tokenizer.SplitMode.A)
        if len(sub_a) == 1:
            sub_tokens_list.append(None)
        elif self.split_mode == 'B':
            sub_tokens_list.append([self._get_dtokens(sub_a, False)])
        else:
            sub_b = token.split(self.tokenizer.SplitMode.B)
            if len(sub_a) == len(sub_b):
                dtokens = self._get_dtokens(sub_a, False)
                sub_tokens_list.append([dtokens, dtokens])
            else:
                sub_tokens_list.append([self._get_dtokens(sub_a, False), self._get_dtokens(sub_b, False)])
    return sub_tokens_list