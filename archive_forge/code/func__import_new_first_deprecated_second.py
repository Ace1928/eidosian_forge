import collections
import dataclasses
import importlib.metadata
import inspect
import logging
import multiprocessing
import os
import sys
import traceback
import types
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple
from importlib.machinery import ModuleSpec
from unittest import mock
import duet
import numpy as np
import pandas as pd
import pytest
import sympy
from _pytest.outcomes import Failed
import cirq.testing
from cirq._compat import (
def _import_new_first_deprecated_second():
    """To ensure that module_a gets initialized only once.

    It is the same as _import_deprecated_first_new_second just with different import order.
    See that for more details.
    """
    from cirq.testing._compat_test_data.module_a import module_b
    assert module_b.MODULE_B_ATTRIBUTE == 'module_b'
    assert 'cirq.testing._compat_test_data.fake_a' not in sys.modules
    assert 'cirq.testing._compat_test_data.module_a' in sys.modules
    from cirq.testing._compat_test_data.fake_a import module_b
    assert 'cirq.testing._compat_test_data.fake_a' in sys.modules
    assert sys.modules['cirq.testing._compat_test_data.fake_a'] == sys.modules['cirq.testing._compat_test_data.module_a']
    assert module_b.MODULE_B_ATTRIBUTE == 'module_b'