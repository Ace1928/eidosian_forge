from __future__ import annotations
import dataclasses
from typing import Optional
from torch.onnx._internal.diagnostics.infra.sarif import (
Contains information that enables a SARIF consumer to locate the external property file that contains the value of an externalized property associated with the run.