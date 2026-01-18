from __future__ import annotations
import contextlib
import gzip
from collections.abc import Generator
from typing import List, Optional
import torch
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
from torch.utils import cpp_backtrace
Dumps the SARIF log to a file.