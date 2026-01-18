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
@dataclasses.dataclass
class MinifierTestResult:
    minifier_code: str
    repro_code: str

    def _get_module(self, t):
        match = re.search('class Repro\\(torch\\.nn\\.Module\\):\\s+([ ].*\\n| *\\n)+', t)
        assert match is not None, 'failed to find module'
        r = match.group(0)
        r = re.sub('\\s+$', '\n', r, flags=re.MULTILINE)
        r = re.sub('\\n{3,}', '\n\n', r)
        return r.strip()

    def minifier_module(self):
        return self._get_module(self.minifier_code)

    def repro_module(self):
        return self._get_module(self.repro_code)