from __future__ import annotations
import dataclasses
from typing import Any, List, Literal, Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
A single artifact. In some cases, this artifact might be nested within another artifact.