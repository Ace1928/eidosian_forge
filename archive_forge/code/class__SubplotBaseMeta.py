from . import _base
from ._axes import *
from ._axes import Axes as Subplot
class _SubplotBaseMeta(type):

    def __instancecheck__(self, obj):
        return isinstance(obj, _base._AxesBase) and obj.get_subplotspec() is not None