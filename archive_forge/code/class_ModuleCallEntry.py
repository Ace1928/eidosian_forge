from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
@dataclass
class ModuleCallEntry:
    fqn: str
    signature: Optional[ModuleCallSignature] = None