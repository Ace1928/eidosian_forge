from collections import namedtuple
from decimal import Decimal
import numpy as np
from . import backends, blas, helpers, parser, paths, sharing
def _get_evaluated_constants(self, backend):
    """Retrieve or generate the cached list of constant operators (mixed
        in with None representing non-consts) and the remaining contraction
        list.
        """
    try:
        return self._evaluated_constants[backend]
    except KeyError:
        self.evaluate_constants(backend)
        return self._evaluated_constants[backend]