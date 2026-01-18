import argparse
import contextlib
import copy
import ctypes
import errno
import functools
import gc
import inspect
import io
import json
import logging
import math
import operator
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from itertools import product, chain
from pathlib import Path
from statistics import mean
from typing import (
from unittest.mock import MagicMock
import expecttest
import numpy as np
import __main__  # type: ignore[import]
import torch
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mps
import torch.backends.xnnpack
import torch.cuda
from torch import Tensor
from torch._C import ScriptDict, ScriptList  # type: ignore[attr-defined]
from torch._utils_internal import get_writable_path
from torch.nn import (
from torch.onnx import (
from torch.testing import make_tensor
from torch.testing._comparison import (
from torch.testing._comparison import not_close_error_metas
from torch.testing._internal.common_dtype import get_all_dtypes
import torch.utils._pytree as pytree
from .composite_compliance import no_dispatch
def _run_with_retry(self, result=None, num_runs_left=0, report_only=True, num_red=0, num_green=0):
    using_unittest = isinstance(result, unittest.TestResult)
    if num_runs_left == 0:
        skipped_msg = {'num_red': num_red, 'num_green': num_green, 'max_num_retries': MAX_NUM_RETRIES, 'rerun_disabled_test': RERUN_DISABLED_TESTS}
        traceback_str = ''
        if RERUN_DISABLED_TESTS and using_unittest:
            if result.failures:
                _, traceback_str = result.failures.pop(-1)
            if result.errors:
                _, traceback_str = result.errors.pop(-1)
            if traceback_str:
                skipped_msg['traceback_str'] = traceback_str
            if num_green == 0:
                result.addSkip(self, json.dumps(skipped_msg))
            if num_red == 0:
                result.addSuccess(self)
        if num_green > 0 and num_red > 0 and using_unittest:
            skipped_msg['flaky'] = True
            result.addSkip(self, json.dumps(skipped_msg))
        return
    if using_unittest:
        failures_before = 0 if result is None else len(result.failures)
        errors_before = 0 if result is None else len(result.errors)
        skipped_before = 0 if result is None else len(result.skipped)
    super_run = super().run
    test_cls = super_run.__self__
    compiled = TEST_WITH_TORCHDYNAMO or TEST_WITH_AOT_EAGER or TEST_WITH_TORCHINDUCTOR
    strict_mode = getattr(test_cls, 'dynamo_strict', False) and compiled
    if strict_mode:
        torch._dynamo.reset()
    if compiled:
        supress_errors = not strict_mode
    else:
        supress_errors = torch._dynamo.config.suppress_errors
    with unittest.mock.patch('torch._dynamo.config.suppress_errors', supress_errors):
        if TEST_WITH_TORCHINDUCTOR:
            super_run = torch._dynamo.optimize('inductor', save_config=False)(super_run)
        elif TEST_WITH_AOT_EAGER:
            super_run = torch._dynamo.optimize('aot_eager_decomp_partition', save_config=False)(super_run)
        elif TEST_WITH_TORCHDYNAMO:
            super_run = torch._dynamo.optimize('eager', save_config=False, nopython=strict_mode)(super_run)
        super_run(result=result)
    if strict_mode:
        torch._dynamo.reset()
    if self._should_stop_test_suite():
        if result.wasSuccessful():
            case = TestCase()
            if TEST_SAVE_XML is not None:
                from xmlrunner.result import _TestInfo
                case = _TestInfo(result, case)
                case.output = _TestInfo.ERROR
                case.elapsed_time = 0.0
                case.test_description = 'TestSuiteEarlyFailure'
            result.failures.append((case, 'TestSuite execution was aborted early'))
            assert result.wasSuccessful() is False
        result.stop()
    if not RETRY_TEST_CASES or not using_unittest:
        return
    err = sys.exc_info()
    num_retries_left = num_runs_left - 1
    if failures_before < len(result.failures):
        print(f'    {self._testMethodName} failed - num_retries_left: {num_retries_left}')
        if report_only and num_retries_left < MAX_NUM_RETRIES or (not report_only and num_retries_left > 0):
            _, traceback_str = result.failures.pop(-1)
            print(traceback_str)
            result.addExpectedFailure(self, err)
        self._run_with_retry(result=result, num_runs_left=num_retries_left, report_only=report_only, num_red=num_red + 1, num_green=num_green)
    elif errors_before < len(result.errors):
        print(f'    {self._testMethodName} errored - num_retries_left: {num_retries_left}')
        if report_only and num_retries_left < MAX_NUM_RETRIES or (not report_only and num_retries_left > 0):
            _, traceback_str = result.errors.pop(-1)
            print(traceback_str)
            result.addExpectedFailure(self, err)
        self._run_with_retry(result=result, num_runs_left=num_retries_left, report_only=report_only, num_red=num_red + 1, num_green=num_green)
    elif RERUN_DISABLED_TESTS and num_retries_left <= MAX_NUM_RETRIES and (skipped_before == len(result.skipped)):
        print(f'    {self._testMethodName} succeeded - num_retries_left: {num_retries_left}')
        result.addSuccess(self)
        self._run_with_retry(result=result, num_runs_left=num_retries_left, report_only=report_only, num_red=num_red, num_green=num_green + 1)
    elif report_only and num_retries_left < MAX_NUM_RETRIES:
        print(f'    {self._testMethodName} succeeded - num_retries_left: {num_retries_left}')
        result.addUnexpectedSuccess(self)
        self._run_with_retry(result=result, num_runs_left=num_retries_left, report_only=report_only, num_red=num_red, num_green=num_green + 1)
    elif not report_only and num_retries_left < MAX_NUM_RETRIES:
        self._run_with_retry(result=result, num_runs_left=0, report_only=report_only, num_red=num_red, num_green=num_green + 1)