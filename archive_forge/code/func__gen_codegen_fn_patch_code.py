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
def _gen_codegen_fn_patch_code(self, device, bug_type):
    assert bug_type in ('compile_error', 'runtime_error', 'accuracy')
    return f'{torch._dynamo.config.codegen_config()}\n{torch._inductor.config.codegen_config()}\ntorch._inductor.config.{('cpp' if device == 'cpu' else 'triton')}.inject_relu_bug_TESTING_ONLY = {bug_type!r}\n'