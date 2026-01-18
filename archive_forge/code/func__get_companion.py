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
def _get_companion(self, data: dict) -> Optional[str]:
    companion = None
    if 'discovery' in data['jupyterlab']:
        if 'server' in data['jupyterlab']['discovery']:
            companion = 'server'
        elif 'kernel' in data['jupyterlab']['discovery']:
            companion = 'kernel'
    return companion