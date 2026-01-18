import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal
def _process_signatures(self, signatures):
    ops = []
    for sig in signatures:
        if isinstance(sig, tuple):
            sig, op = sig
        else:
            op = self._default_func
        ins, outs = self._sig_str_to_tuple(sig)
        if len(ins) != self._nin:
            raise ValueError(f'signature {sig} for dtypes is invalid number of inputs is not consistent with general signature')
        if len(outs) != self._nout:
            raise ValueError(f'signature {sig} for dtypes is invalid number of inputs is not consistent with general signature')
        ops.append(_OpsRegister._Op(ins, outs, op))
    return ops