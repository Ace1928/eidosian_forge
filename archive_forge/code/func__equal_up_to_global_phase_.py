import numpy as np
import cirq
def _equal_up_to_global_phase_(self, other, atol):
    if not isinstance(self.val[0], type(other)):
        return NotImplemented
    return cirq.equal_up_to_global_phase(self.val[0], other, atol=atol)