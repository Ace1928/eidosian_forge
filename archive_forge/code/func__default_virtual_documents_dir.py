import asyncio
import os
import sys
import traceback
from typing import Dict, Text, Tuple, cast
from jupyter_core.paths import jupyter_config_path
from jupyter_server.services.config import ConfigManager
from traitlets import Bool
from traitlets import Dict as Dict_
from traitlets import Instance
from traitlets import List as List_
from traitlets import Unicode, default
from .constants import (
from .schema import LANGUAGE_SERVER_SPEC_MAP
from .session import LanguageServerSession
from .trait_types import LoadableCallable, Schema
from .types import (
@default('virtual_documents_dir')
def _default_virtual_documents_dir(self):
    return os.getenv('JP_LSP_VIRTUAL_DIR', '.virtual_documents')