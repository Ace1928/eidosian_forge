from __future__ import annotations
import dataclasses
from typing import Literal, Optional
from torch.onnx._internal.diagnostics.infra.sarif import _property_bag
Information about a rule or notification that can be configured at runtime.