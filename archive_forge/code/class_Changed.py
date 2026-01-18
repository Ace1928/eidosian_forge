from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from click import style
from black.output import err, out
class Changed(Enum):
    NO = 0
    CACHED = 1
    YES = 2