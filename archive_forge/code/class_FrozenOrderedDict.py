import functools
import operator
import sys
import six
from six.moves import reload_module
class FrozenOrderedDict(frozendict):
    """
    A frozendict subclass that maintains key order
    """
    dict_cls = OrderedDict