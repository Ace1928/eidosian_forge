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
def _test_deprecate_attributes_assert_attributes_in_sys_modules():
    """Ensure submodule attributes are consistent with sys.modules items."""
    import cirq.testing._compat_test_data.module_a as module_a0
    module_a1 = deprecate_attributes('cirq.testing._compat_test_data.module_a', {'MODULE_A_ATTRIBUTE': ('v0.6', 'use plain string instead')})
    assert module_a1 is not module_a0
    assert module_a1 is cirq.testing._compat_test_data.module_a
    assert module_a1 is sys.modules['cirq.testing._compat_test_data.module_a']