from __future__ import annotations
import dataclasses
from typing import List, Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
Describes how a converter transformed the output of a static analysis tool from the analysis tool's native output format into the SARIF format.