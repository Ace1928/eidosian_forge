from __future__ import annotations
import dataclasses
from typing import Any, List, Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
Describes a sequence of code locations that specify a path through a single thread of execution such as an operating system or fiber.