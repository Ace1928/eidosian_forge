import numpy as np
import pytest
import cirq
class Yes3(EmptyOp):

    def _has_unitary_(self):
        return NotImplemented

    def _decompose_(self):
        return NotImplemented

    def _apply_unitary_(self, args):
        return args.target_tensor

    def _unitary_(self):
        assert False