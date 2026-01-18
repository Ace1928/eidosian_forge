from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
@dataclass
class ExportedProgram:
    graph_module: GraphModule
    opset_version: Dict[str, int]
    range_constraints: Dict[str, RangeConstraint]
    schema_version: int
    dialect: str