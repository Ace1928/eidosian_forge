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
class DisabledPipes(list):
    """Manager for temporary pipeline disabling."""

    def __init__(self, nlp: Language, names: List[str]) -> None:
        self.nlp = nlp
        self.names = names
        for name in self.names:
            self.nlp.disable_pipe(name)
        list.__init__(self)
        self.extend(self.names)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.restore()

    def restore(self) -> None:
        """Restore the pipeline to its state when DisabledPipes was created."""
        for name in self.names:
            if name not in self.nlp.component_names:
                raise ValueError(Errors.E008.format(name=name))
            self.nlp.enable_pipe(name)
        self[:] = []