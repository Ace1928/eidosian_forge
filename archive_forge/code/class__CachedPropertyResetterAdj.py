from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.reportviews import DegreeView, EdgeView, NodeView
from networkx.exception import NetworkXError
class _CachedPropertyResetterAdj:
    """Data Descriptor class for _adj that resets ``adj`` cached_property when needed

    This assumes that the ``cached_property`` ``G.adj`` should be reset whenever
    ``G._adj`` is set to a new value.

    This object sits on a class and ensures that any instance of that
    class clears its cached property "adj" whenever the underlying
    instance attribute "_adj" is set to a new object. It only affects
    the set process of the obj._adj attribute. All get/del operations
    act as they normally would.

    For info on Data Descriptors see: https://docs.python.org/3/howto/descriptor.html
    """

    def __set__(self, obj, value):
        od = obj.__dict__
        od['_adj'] = value
        if 'adj' in od:
            del od['adj']