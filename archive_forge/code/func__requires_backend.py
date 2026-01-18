from __future__ import annotations
import contextvars
import functools
import itertools
import sys
import uuid
import warnings
from collections.abc import Generator, Callable, Iterable
from contextlib import contextmanager
from functools import singledispatchmethod
from typing import TypeVar, ContextManager, TypedDict, Union, Optional, Dict
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse import (
from qiskit.providers.backend import BackendV2
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.alignments import AlignmentKind
def _requires_backend(function: Callable[..., T]) -> Callable[..., T]:
    """Decorator a function to raise if it is called without a builder with a
    set backend.
    """

    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        if self.backend is None:
            raise exceptions.BackendNotSet('This function requires the builder to have a "backend" set.')
        return function(self, *args, **kwargs)
    return wrapper