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
def _get_dtokens(self, sudachipy_tokens, need_sub_tokens: bool=True):
    sub_tokens_list = self._get_sub_tokens(sudachipy_tokens) if need_sub_tokens else None
    dtokens = [DetailedToken(token.surface(), '-'.join([xx for xx in token.part_of_speech()[:4] if xx != '*']), ';'.join([xx for xx in token.part_of_speech()[4:] if xx != '*']), token.dictionary_form(), token.normalized_form(), token.reading_form(), sub_tokens_list[idx] if sub_tokens_list else None) for idx, token in enumerate(sudachipy_tokens) if len(token.surface()) > 0]
    return [t for idx, t in enumerate(dtokens) if idx == 0 or not t.surface.isspace() or t.tag != '空白' or (not dtokens[idx - 1].surface.isspace()) or (dtokens[idx - 1].tag != '空白')]