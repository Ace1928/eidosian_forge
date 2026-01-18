from pyomo.common.autoslots import AutoSlots
from pyomo.core.base.block import _BlockData, IndexedBlock
from pyomo.core.base.global_set import UnindexedComponent_index, UnindexedComponent_set
class _TransformedDisjunctData(_BlockData):
    __slots__ = ('_src_disjunct',)
    __autoslot_mappers__ = {'_src_disjunct': AutoSlots.weakref_mapper}

    @property
    def src_disjunct(self):
        return None if self._src_disjunct is None else self._src_disjunct()

    def __init__(self, component):
        _BlockData.__init__(self, component)
        self._src_disjunct = None