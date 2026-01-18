from __future__ import annotations
import dataclasses
from typing import List, Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
A set of threadFlows which together describe a pattern of code execution relevant to detecting a result.