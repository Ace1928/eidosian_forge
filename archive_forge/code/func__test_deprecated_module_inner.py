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
def _test_deprecated_module_inner(outdated_method, deprecation_messages):
    import cirq
    with cirq.testing.assert_logs('init:compat_test_data', 'init:module_a', min_level=logging.INFO, max_level=logging.INFO, count=2):
        with cirq.testing.assert_deprecated(*[msg for dep in deprecation_messages for msg in dep], deadline='v0.20', count=len(deprecation_messages)):
            warnings.simplefilter('always')
            outdated_method()