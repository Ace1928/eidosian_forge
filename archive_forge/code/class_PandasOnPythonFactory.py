import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@doc(_doc_factory_class, execution_name='PandasOnPython')
class PandasOnPythonFactory(BaseFactory):

    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name='``PandasOnPythonIO``')
    def prepare(cls):
        from modin.core.execution.python.implementations.pandas_on_python.io import PandasOnPythonIO
        cls.io_cls = PandasOnPythonIO