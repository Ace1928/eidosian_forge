from __future__ import annotations
import dataclasses
from typing import List, Literal, Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
Describes a condition relevant to the tool itself, as opposed to being relevant to a target being analyzed by the tool.