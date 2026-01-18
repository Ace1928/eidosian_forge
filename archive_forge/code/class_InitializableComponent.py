from typing import (
from thinc.api import Model, Optimizer
from .compat import Protocol, runtime_checkable
@runtime_checkable
class InitializableComponent(Protocol):

    def initialize(self, get_examples: Callable[[], Iterable['Example']], nlp: 'Language', **kwargs: Any):
        ...