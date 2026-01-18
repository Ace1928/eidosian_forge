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
def _test_broken_module_3_inner():
    import cirq.testing._compat_test_data
    warnings.simplefilter('always')
    with cirq.testing.assert_deprecated(deadline='v0.20', count=None):
        with pytest.raises(DeprecatedModuleImportError, match='missing_module cannot be imported. The typical reasons'):
            cirq.testing._compat_test_data.broken_ref.something()