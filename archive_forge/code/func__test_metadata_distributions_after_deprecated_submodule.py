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
def _test_metadata_distributions_after_deprecated_submodule():
    deprecated_submodule(new_module_name='cirq.neutral_atoms', old_parent='cirq', old_child='swiss_atoms', deadline='v0.14', create_attribute=True)
    m = pytest.importorskip('importlib_metadata')
    distlist = list(m.distributions())
    assert all((isinstance(d.name, str) for d in distlist))