import tempfile
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
import srsly
from ... import util
from ...errors import Errors, Warnings
from ...language import BaseDefaults, Language
from ...scorer import Scorer
from ...tokens import Doc
from ...training import Example, validate_examples
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
class ChineseDefaults(BaseDefaults):
    config = load_config_from_str(DEFAULT_CONFIG)
    lex_attr_getters = LEX_ATTRS
    stop_words = STOP_WORDS
    writing_system = {'direction': 'ltr', 'has_case': False, 'has_letters': False}