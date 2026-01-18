from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, cast
import numpy
from thinc.api import (
from thinc.loss import Loss
from thinc.types import Floats2d, Ints1d
from ...attrs import ID, ORTH
from ...errors import Errors
from ...util import OOV_RANK, registry
from ...vectors import Mode as VectorsMode
class _RandomWords:

    def __init__(self, vocab: 'Vocab') -> None:
        self.words = [lex.text for lex in vocab if lex.prob != 0.0]
        self.words = self.words[:10000]
        probs = [lex.prob for lex in vocab if lex.prob != 0.0]
        probs = probs[:10000]
        probs: numpy.ndarray = numpy.exp(numpy.array(probs, dtype='f'))
        probs /= probs.sum()
        self.probs = probs
        self._cache: List[int] = []

    def next(self) -> str:
        if not self._cache:
            self._cache.extend(numpy.random.choice(len(self.words), 10000, p=self.probs))
        index = self._cache.pop()
        return self.words[index]