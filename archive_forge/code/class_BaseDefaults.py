import functools
import inspect
import itertools
import multiprocessing as mp
import random
import traceback
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain, cycle
from pathlib import Path
from timeit import default_timer as timer
from typing import (
import srsly
from thinc.api import Config, CupyOps, Optimizer, get_current_ops
from . import about, ty, util
from .compat import Literal
from .errors import Errors, Warnings
from .git_info import GIT_VERSION
from .lang.punctuation import TOKENIZER_INFIXES, TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES
from .lang.tokenizer_exceptions import BASE_EXCEPTIONS, URL_MATCH
from .lookups import load_lookups
from .pipe_analysis import analyze_pipes, print_pipe_analysis, validate_attrs
from .schemas import (
from .scorer import Scorer
from .tokenizer import Tokenizer
from .tokens import Doc
from .tokens.underscore import Underscore
from .training import Example, validate_examples
from .training.initialize import init_tok2vec, init_vocab
from .util import (
from .vectors import BaseVectors
from .vocab import Vocab, create_vocab
class BaseDefaults:
    """Language data defaults, available via Language.Defaults. Can be
    overwritten by language subclasses by defining their own subclasses of
    Language.Defaults.
    """
    config: Config = Config(section_order=CONFIG_SECTION_ORDER)
    tokenizer_exceptions: Dict[str, List[dict]] = BASE_EXCEPTIONS
    prefixes: Optional[Sequence[Union[str, Pattern]]] = TOKENIZER_PREFIXES
    suffixes: Optional[Sequence[Union[str, Pattern]]] = TOKENIZER_SUFFIXES
    infixes: Optional[Sequence[Union[str, Pattern]]] = TOKENIZER_INFIXES
    token_match: Optional[Callable] = None
    url_match: Optional[Callable] = URL_MATCH
    syntax_iterators: Dict[str, Callable] = {}
    lex_attr_getters: Dict[int, Callable[[str], Any]] = {}
    stop_words: Set[str] = set()
    writing_system = {'direction': 'ltr', 'has_case': True, 'has_letters': True}