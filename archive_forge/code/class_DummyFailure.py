from __future__ import annotations
import pytest
from xarray.backends.common import robust_getitem
class DummyFailure(Exception):
    pass