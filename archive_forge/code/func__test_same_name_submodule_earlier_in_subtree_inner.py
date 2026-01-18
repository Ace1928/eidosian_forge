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
def _test_same_name_submodule_earlier_in_subtree_inner():
    from cirq.testing._compat_test_data.module_a.sub.subsub.dupe import DUPE_CONSTANT
    assert DUPE_CONSTANT