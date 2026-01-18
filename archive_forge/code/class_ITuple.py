from zope.interface import classImplements
from zope.interface.common import collections
from zope.interface.common import io
from zope.interface.common import numbers
class ITuple(collections.ISequence):
    """
    Interface for :class:`tuple`
    """
    extra_classes = (tuple,)