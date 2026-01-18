import json
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union
import tornado
from jupyterlab_server.translation_utils import translator
from traitlets import Enum
from traitlets.config import Configurable, LoggingConfigurable
from jupyterlab.commands import (
@dataclass(frozen=True)
class ActionResult:
    """Action result

    Attributes:
        status: Action status - ["ok", "warning", "error"]
        message: Action status explanation
        needs_restart: Required action follow-up - Valid follow-up are "frontend", "kernel" and "server"
    """
    status: str
    message: Optional[str] = None
    needs_restart: List[str] = field(default_factory=list)