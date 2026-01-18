from modin.core.execution.python.implementations.pandas_on_python.dataframe.dataframe import (
from modin.core.io import BaseIO
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
class PandasOnPythonIO(BaseIO):
    """
    Class for storing IO functions operating on pandas storage format and Python engine.

    Inherits default function implementations from ``BaseIO`` parent class.
    """
    frame_cls = PandasOnPythonDataframe
    query_compiler_cls = PandasQueryCompiler