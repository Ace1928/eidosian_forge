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
@default('conf_d_language_servers')
def _default_conf_d_language_servers(self) -> KeyedLanguageServerSpecs:
    language_servers: KeyedLanguageServerSpecs = {}
    manager = ConfigManager(read_config_path=jupyter_config_path())
    for app in APP_CONFIG_D_SECTIONS:
        language_servers.update(**manager.get(f'jupyter{app}config').get(self.__class__.__name__, {}).get('language_servers', {}))
    return language_servers