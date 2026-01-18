from zope.interface import Interface
from zope.interface.common import collections
class IReadSequence(collections.IContainer, IFiniteSequence):
    """
    read interface shared by tuple and list

    This interface is similar to
    :class:`~zope.interface.common.collections.ISequence`, but
    requires that all instances be totally ordered. Most users
    should prefer ``ISequence``.

    .. versionchanged:: 5.0.0
       Extend ``IContainer``
    """

    def __contains__(item):
        """``x.__contains__(item) <==> item in x``"""

    def __lt__(other):
        """``x.__lt__(other) <==> x < other``"""

    def __le__(other):
        """``x.__le__(other) <==> x <= other``"""

    def __eq__(other):
        """``x.__eq__(other) <==> x == other``"""

    def __ne__(other):
        """``x.__ne__(other) <==> x != other``"""

    def __gt__(other):
        """``x.__gt__(other) <==> x > other``"""

    def __ge__(other):
        """``x.__ge__(other) <==> x >= other``"""

    def __add__(other):
        """``x.__add__(other) <==> x + other``"""

    def __mul__(n):
        """``x.__mul__(n) <==> x * n``"""

    def __rmul__(n):
        """``x.__rmul__(n) <==> n * x``"""