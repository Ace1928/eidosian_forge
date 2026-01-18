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
def _gen_test_code(self, run_code, repro_after, repro_level):
    return f'import torch\nimport torch._dynamo\n{torch._dynamo.config.codegen_config()}\n{torch._inductor.config.codegen_config()}\ntorch._dynamo.config.repro_after = "{repro_after}"\ntorch._dynamo.config.repro_level = {repro_level}\ntorch._dynamo.config.debug_dir_root = "{self.DEBUG_DIR}"\n{run_code}\n'