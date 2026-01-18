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
@dataclass
class ExtensionsCache:
    """Extensions cache

    Attributes:
        cache: Extension list per page
        last_page: Last available page result
    """
    cache: Dict[int, Optional[Dict[str, ExtensionPackage]]] = field(default_factory=dict)
    last_page: int = 1