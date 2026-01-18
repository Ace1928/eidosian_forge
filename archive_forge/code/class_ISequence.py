from zope.interface import Interface
from zope.interface.common import collections
class ISequence(IReadSequence, IWriteSequence):
    """
    Full sequence contract.

    New code should prefer
    :class:`~zope.interface.common.collections.IMutableSequence`.

    Compared to that interface, which is implemented by :class:`list`
    (:class:`~zope.interface.common.builtins.IList`), among others,
    this interface is missing the following methods:

        - clear

        - count

        - index

    This interface adds the following methods:

        - sort
    """