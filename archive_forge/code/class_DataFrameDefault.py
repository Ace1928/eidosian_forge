import pandas
from modin.utils import _inherit_docstrings
from .default import DefaultMethod
@_inherit_docstrings(DefaultMethod)
class DataFrameDefault(DefaultMethod):
    DEFAULT_OBJECT_TYPE = pandas.DataFrame