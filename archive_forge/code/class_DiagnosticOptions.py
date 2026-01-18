from __future__ import annotations
import dataclasses
import enum
import logging
from typing import FrozenSet, List, Mapping, Optional, Sequence, Tuple
from torch.onnx._internal.diagnostics.infra import formatter, sarif
@dataclasses.dataclass
class DiagnosticOptions:
    """Options for diagnostic context.

    Attributes:
        verbosity_level: Set the amount of information logged for each diagnostics,
            equivalent to the 'level' in Python logging module.
        warnings_as_errors: When True, warning diagnostics are treated as error diagnostics.
    """
    verbosity_level: int = dataclasses.field(default=logging.INFO)
    "Set the amount of information logged for each diagnostics, equivalent to the 'level' in Python logging module."
    warnings_as_errors: bool = dataclasses.field(default=False)
    'If True, warning diagnostics are treated as error diagnostics.'