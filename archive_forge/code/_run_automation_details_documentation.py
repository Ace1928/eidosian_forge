from __future__ import annotations
import dataclasses
from typing import Optional
from torch.onnx._internal.diagnostics.infra.sarif import _message, _property_bag
Information that describes a run's identity and role within an engineering system process.