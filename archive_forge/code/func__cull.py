import operator
import warnings
import dask
from dask import core
from dask.core import istask
from dask.dataframe.core import _concat
from dask.dataframe.optimize import optimize
from dask.dataframe.shuffle import shuffle_group
from dask.highlevelgraph import HighLevelGraph
from .scheduler import MultipleReturnFunc, multiple_return_get
def _cull(self, parts_out):
    return MultipleReturnSimpleShuffleLayer(self.name, self.column, self.npartitions, self.npartitions_input, self.ignore_index, self.name_input, self.meta_input, parts_out=parts_out)