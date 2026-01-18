from collections import namedtuple
from decimal import Decimal
import numpy as np
from . import backends, blas, helpers, parser, paths, sharing
def _contract_with_conversion(self, arrays, out, backend, evaluate_constants=False):
    """Special contraction, i.e., contraction with a different backend
        but converting to and from that backend. Retrieves or generates a
        cached expression using ``arrays`` as templates, then calls it
        with ``arrays``.

        If ``evaluate_constants=True``, perform a partial contraction that
        prepares the constant tensors and operations with the right backend.
        """
    if evaluate_constants:
        return backends.evaluate_constants(backend, arrays, self)
    result = self._get_backend_expression(arrays, backend)(*arrays)
    if out is not None:
        out[()] = result
        return out
    return result