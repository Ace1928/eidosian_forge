import dataclasses
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Optional
from unittest.mock import patch
import torch
import torch._dynamo
import torch._dynamo.test_case
from torch.utils._traceback import report_compile_source_on_error
import torch
import torch._dynamo
def _run_full_test(self, run_code, repro_after, expected_error, *, isolate, minifier_args=()) -> Optional[MinifierTestResult]:
    if isolate:
        repro_level = 3
    elif expected_error is None or expected_error == 'AccuracyError':
        repro_level = 4
    else:
        repro_level = 2
    test_code = self._gen_test_code(run_code, repro_after, repro_level)
    print('running test', file=sys.stderr)
    test_proc, repro_dir = self._run_test_code(test_code, isolate=isolate)
    if expected_error is None:
        self.assertEqual(test_proc.returncode, 0)
        self.assertIsNone(repro_dir)
        return None
    self.assertIn(expected_error, test_proc.stderr.decode('utf-8'))
    self.assertIsNotNone(repro_dir)
    print('running minifier', file=sys.stderr)
    minifier_proc, minifier_code = self._run_minifier_launcher(repro_dir, isolate=isolate, minifier_args=minifier_args)
    print('running repro', file=sys.stderr)
    repro_proc, repro_code = self._run_repro(repro_dir, isolate=isolate)
    self.assertIn(expected_error, repro_proc.stderr.decode('utf-8'))
    self.assertNotEqual(repro_proc.returncode, 0)
    return MinifierTestResult(minifier_code=minifier_code, repro_code=repro_code)