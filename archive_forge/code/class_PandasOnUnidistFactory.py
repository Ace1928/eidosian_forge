import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@doc(_doc_factory_class, execution_name='PandasOnUnidist')
class PandasOnUnidistFactory(BaseFactory):

    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name='``PandasOnUnidistIO``')
    def prepare(cls):
        from modin.core.execution.unidist.implementations.pandas_on_unidist.io import PandasOnUnidistIO
        cls.io_cls = PandasOnUnidistIO