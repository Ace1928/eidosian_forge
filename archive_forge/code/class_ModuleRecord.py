import dataclasses
from dataclasses import field
from types import CodeType, ModuleType
from typing import Any, Dict
@dataclasses.dataclass
class ModuleRecord:
    module: ModuleType
    accessed_attrs: Dict[str, Any] = field(default_factory=dict)