import inspect
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
@dataclass(frozen=True)
class ExportCase:
    example_inputs: InputsType
    description: str
    model: torch.nn.Module
    name: str
    extra_inputs: Optional[InputsType] = None
    tags: Set[str] = field(default_factory=set)
    support_level: SupportLevel = SupportLevel.SUPPORTED
    dynamic_shapes: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        check_inputs_type(self.example_inputs)
        if self.extra_inputs is not None:
            check_inputs_type(self.extra_inputs)
        for tag in self.tags:
            _validate_tag(tag)
        if not isinstance(self.description, str) or len(self.description) == 0:
            raise ValueError(f'Invalid description: "{self.description}"')