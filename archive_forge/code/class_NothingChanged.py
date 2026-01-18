from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from click import style
from black.output import err, out
class NothingChanged(UserWarning):
    """Raised when reformatted code is the same as source."""