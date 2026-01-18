from __future__ import annotations
import dataclasses
from typing import List, Optional
from torch.onnx._internal.diagnostics.infra.sarif import _message, _property_bag
Information about the relation of one location to another.