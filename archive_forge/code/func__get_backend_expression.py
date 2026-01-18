from collections import namedtuple
from decimal import Decimal
import numpy as np
from . import backends, blas, helpers, parser, paths, sharing
def _get_backend_expression(self, arrays, backend):
    try:
        return self._backend_expressions[backend]
    except KeyError:
        fn = backends.build_expression(backend, arrays, self)
        self._backend_expressions[backend] = fn
        return fn