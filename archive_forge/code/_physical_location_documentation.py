from __future__ import annotations
import dataclasses
from typing import Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
A physical location relevant to a result. Specifies a reference to a programming artifact together with a range of bytes or characters within that artifact.