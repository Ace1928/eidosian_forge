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
def _ensure_compat_errors(info, app_options):
    """Ensure that the app info has compat_errors field"""
    handler = _AppHandler(app_options)
    info['compat_errors'] = handler._get_extension_compat()