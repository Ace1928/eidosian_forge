from zope.interface import classImplements
from zope.interface.common import collections
from zope.interface.common import io
from zope.interface.common import numbers
class IFile(io.IIOBase):
    """
    Interface for :class:`file`.

    It is recommended to use the interfaces from :mod:`zope.interface.common.io`
    instead of this interface.

    On Python 3, there is no single implementation of this interface;
    depending on the arguments, the :func:`open` builtin can return
    many different classes that implement different interfaces from
    :mod:`zope.interface.common.io`.
    """
    extra_classes = ()