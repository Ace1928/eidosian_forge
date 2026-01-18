from __future__ import annotations
import dataclasses
from typing import List, Optional
from torch.onnx._internal.diagnostics.infra.sarif import _location, _property_bag
A function call within a stack trace.