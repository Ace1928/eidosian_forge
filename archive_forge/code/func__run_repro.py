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
def _run_repro(self, repro_dir, *, isolate=True):
    self.assertIsNotNone(repro_dir)
    repro_file = os.path.join(repro_dir, 'repro.py')
    with open(repro_file) as f:
        repro_code = f.read()
    self.assertTrue(os.path.exists(repro_file))
    repro_proc = self._maybe_subprocess_run(['python3', repro_file], isolate=isolate, cwd=repro_dir)
    print('repro stdout:', repro_proc.stdout.decode('utf-8'))
    print('repro stderr:', repro_proc.stderr.decode('utf-8'))
    return (repro_proc, repro_code)