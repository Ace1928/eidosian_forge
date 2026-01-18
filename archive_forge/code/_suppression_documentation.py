from __future__ import annotations
import dataclasses
from typing import Literal, Optional
from torch.onnx._internal.diagnostics.infra.sarif import _location, _property_bag
A suppression that is relevant to a result.