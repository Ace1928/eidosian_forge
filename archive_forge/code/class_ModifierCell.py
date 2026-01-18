from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
class ModifierCell(HybridRecurrentCell):
    """Base class for modifier cells. A modifier
    cell takes a base cell, apply modifications
    on it (e.g. Zoneout), and returns a new cell.

    After applying modifiers the base cell should
    no longer be called directly. The modifier cell
    should be used instead.
    """

    def __init__(self, base_cell):
        assert not base_cell._modified, 'Cell %s is already modified. One cell cannot be modified twice' % base_cell.name
        base_cell._modified = True
        super(ModifierCell, self).__init__(prefix=base_cell.prefix + self._alias(), params=None)
        self.base_cell = base_cell

    @property
    def params(self):
        return self.base_cell.params

    def state_info(self, batch_size=0):
        return self.base_cell.state_info(batch_size)

    def begin_state(self, func=symbol.zeros, **kwargs):
        assert not self._modified, 'After applying modifier cells (e.g. DropoutCell) the base cell cannot be called directly. Call the modifier cell instead.'
        self.base_cell._modified = False
        begin = self.base_cell.begin_state(func=func, **kwargs)
        self.base_cell._modified = True
        return begin

    def hybrid_forward(self, F, inputs, states):
        raise NotImplementedError

    def __repr__(self):
        s = '{name}({base_cell})'
        return s.format(name=self.__class__.__name__, **self.__dict__)