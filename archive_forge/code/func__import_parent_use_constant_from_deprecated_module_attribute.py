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
def _import_parent_use_constant_from_deprecated_module_attribute():
    """to ensure that module initializations set attributes correctly"""
    import cirq.testing._compat_test_data
    assert cirq.testing._compat_test_data.fake_a.DUPE_CONSTANT is False
    assert 'module_a for module deprecation tests' in cirq.testing._compat_test_data.fake_a.__doc__
    assert 'Test module for deprecation testing' in cirq.testing._compat_test_data.__doc__