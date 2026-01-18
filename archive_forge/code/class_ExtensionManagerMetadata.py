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
class ExtensionManagerMetadata:
    """Extension manager metadata.

    Attributes:
        name: Extension manager name to be displayed
        can_install: Whether the extension manager can un-/install packages (default False)
        install_path: Installation path for the extensions (default None); e.g. environment path
    """
    name: str
    can_install: bool = False
    install_path: Optional[str] = None