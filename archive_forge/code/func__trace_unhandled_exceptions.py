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
def _trace_unhandled_exceptions(*args, queue: 'multiprocessing.Queue', func: Callable, **kwargs):
    try:
        func(*args, **kwargs)
        queue.put(None)
    except BaseException as ex:
        msg = str(ex)
        queue.put((type(ex).__name__, msg, traceback.format_exc()))