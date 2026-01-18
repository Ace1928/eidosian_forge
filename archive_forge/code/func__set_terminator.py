import contextlib
import functools
from llvmlite.ir import instructions, types, values
def _set_terminator(self, term):
    assert not self.block.is_terminated
    self._insert(term)
    self.block.terminator = term
    return term