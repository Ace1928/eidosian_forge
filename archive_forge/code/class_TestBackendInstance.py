from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pytest
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
class TestBackendInstance:

    def test_get_backend(self):
        args = ({1: 0, 2: 2}, {-1: 1, 3: 1}, {3: 0, -1: 1}, 2, 4)
        backend = CanonBackend.get_backend(s.SCIPY_CANON_BACKEND, *args)
        assert isinstance(backend, SciPyCanonBackend)
        backend = CanonBackend.get_backend(s.NUMPY_CANON_BACKEND, *args)
        assert isinstance(backend, NumPyCanonBackend)
        with pytest.raises(KeyError):
            CanonBackend.get_backend('notabackend')