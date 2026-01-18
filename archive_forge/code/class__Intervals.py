from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast
import numpy
from thinc.api import Config, Model, Ops, Optimizer, get_current_ops, set_dropout_rate
from thinc.types import Floats2d, Ints1d, Ints2d, Ragged
from ..compat import Protocol, runtime_checkable
from ..errors import Errors
from ..language import Language
from ..scorer import Scorer
from ..tokens import Doc, Span, SpanGroup
from ..training import Example, validate_examples
from ..util import registry
from ..vocab import Vocab
from .trainable_pipe import TrainablePipe
@dataclass
class _Intervals:
    """
    Helper class to avoid storing overlapping spans.
    """

    def __init__(self):
        self.ranges = set()

    def add(self, i, j):
        for e in range(i, j):
            self.ranges.add(e)

    def __contains__(self, rang):
        i, j = rang
        for e in range(i, j):
            if e in self.ranges:
                return True
        return False