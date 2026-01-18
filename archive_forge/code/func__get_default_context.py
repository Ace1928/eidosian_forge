import dataclasses
import inspect
import enum
import functools
import textwrap
from typing import (
from typing_extensions import Protocol
from cirq import circuits
def _get_default_context(func: TRANSFORMER) -> TransformerContext:
    sig = inspect.signature(func)
    default_context = sig.parameters['context'].default
    assert default_context != inspect.Parameter.empty, '`context` argument must have a default value specified.'
    return default_context