import operator
import sys
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
import numpy as np
import pandas as pd
import param
from ..core.data import PandasInterface
from ..core.dimension import Dimension
from ..core.util import flatten, resolve_dependent_value, unique_iterator
class df_dim(dim):
    """
    A subclass of dim which provides access to the DataFrame namespace
    along with tab-completion and type coercion allowing the expression
    to be applied on any columnar dataset.
    """
    namespace = 'dataframe'
    _accessor = 'pd'

    def __init__(self, obj, *args, **kwargs):
        super().__init__(obj, *args, **kwargs)
        self._ns = pd.Series

    def interface_applies(self, dataset, coerce):
        return not dataset.interface.gridded and (coerce or isinstance(dataset.interface, PandasInterface))

    def _compute_data(self, data, drop_index, compute):
        if hasattr(data, 'compute') and compute:
            data = data.compute()
        if not drop_index:
            return data
        if compute and hasattr(data, 'to_numpy'):
            return data.to_numpy()
        return data.values

    def _coerce(self, dataset):
        if self.interface_applies(dataset, coerce=False):
            return dataset
        pandas_interfaces = param.concrete_descendents(PandasInterface)
        datatypes = [intfc.datatype for intfc in pandas_interfaces.values() if dataset.interface.multi == intfc.multi]
        return dataset.clone(datatype=datatypes)

    @property
    def loc(self):
        return loc(self)