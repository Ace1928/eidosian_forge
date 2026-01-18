from contextlib import contextmanager
import pytest
import modin.pandas as pd
from modin import set_execution
from modin.config import Engine, Parameter, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.core.execution.dispatching.factories.dispatcher import (
from modin.core.execution.python.implementations.pandas_on_python.io import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
@contextmanager
def _switch_value(config: Parameter, value: str):
    old_value = config.get()
    try:
        yield config.put(value)
    finally:
        config.put(old_value)