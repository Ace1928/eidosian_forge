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
def _switch_execution(engine: str, storage_format: str):
    old_engine, old_storage = set_execution(engine, storage_format)
    try:
        yield
    finally:
        set_execution(old_engine, old_storage)