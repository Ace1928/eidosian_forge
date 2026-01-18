import operator
import typing
from collections.abc import MappingView, MutableMapping, MutableSet
class _GlobalPhaseSentinel:
    __slots__ = ()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self

    def __reduce__(self):
        return (operator.attrgetter('GLOBAL_PHASE'), (ParameterTable,))

    def __repr__(self):
        return '<global-phase sentinel>'