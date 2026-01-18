from zope.interface import classImplements
from zope.interface.common import collections
from zope.interface.common import io
from zope.interface.common import numbers
class ITextString(collections.ISequence):
    """
    Interface for text ("unicode") strings.

    This is :class:`str`
    """
    extra_classes = (str,)