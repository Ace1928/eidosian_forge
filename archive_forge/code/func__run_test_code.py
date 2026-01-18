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
def _run_test_code(self, code, *, isolate):
    proc = self._maybe_subprocess_run(['python3', '-c', code], isolate=isolate, cwd=self.DEBUG_DIR)
    print('test stdout:', proc.stdout.decode('utf-8'))
    print('test stderr:', proc.stderr.decode('utf-8'))
    repro_dir_match = re.search('(\\S+)minifier_launcher.py', proc.stderr.decode('utf-8'))
    if repro_dir_match is not None:
        return (proc, repro_dir_match.group(1))
    return (proc, None)