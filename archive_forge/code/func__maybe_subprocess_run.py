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
def _maybe_subprocess_run(self, args, *, isolate, cwd=None):
    if not isolate:
        assert len(args) >= 2, args
        assert args[0] == 'python3', args
        if args[1] == '-c':
            assert len(args) == 3, args
            code = args[2]
            args = ['-c']
        else:
            assert len(args) >= 2, args
            with open(args[1]) as f:
                code = f.read()
            args = args[1:]
        dynamo_config = torch._dynamo.config.shallow_copy_dict()
        inductor_config = torch._inductor.config.shallow_copy_dict()
        try:
            stderr = io.StringIO()
            log_handler = logging.StreamHandler(stderr)
            log = logging.getLogger('torch._dynamo')
            log.addHandler(log_handler)
            try:
                prev_cwd = os.getcwd()
                if cwd is not None:
                    os.chdir(cwd)
                with patch('sys.argv', args), report_compile_source_on_error():
                    exec(code, {'__name__': '__main__', '__compile_source__': code})
                rc = 0
            except Exception:
                rc = 1
                traceback.print_exc(file=stderr)
            finally:
                log.removeHandler(log_handler)
                if cwd is not None:
                    os.chdir(prev_cwd)
                torch._dynamo.reset()
        finally:
            torch._dynamo.config.load_config(dynamo_config)
            torch._inductor.config.load_config(inductor_config)
        return subprocess.CompletedProcess(args, rc, b'', stderr.getvalue().encode('utf-8'))
    else:
        return subprocess.run(args, capture_output=True, cwd=cwd, check=False)