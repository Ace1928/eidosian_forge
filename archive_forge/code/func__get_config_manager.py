from __future__ import annotations
import json
import os.path as osp
from glob import iglob
from itertools import chain
from logging import Logger
from os.path import join as pjoin
from typing import Any
import json5
from jupyter_core.paths import SYSTEM_CONFIG_PATH, jupyter_config_dir, jupyter_path
from jupyter_server.services.config.manager import ConfigManager, recursive_update
from jupyter_server.utils import url_path_join as ujoin
from traitlets import Bool, HasTraits, List, Unicode, default
def _get_config_manager(level: str) -> ConfigManager:
    """Get the location of config files for the current context
    Returns the string to the environment
    """
    allowed = ['all', 'user', 'sys_prefix', 'system', 'app', 'extension']
    if level not in allowed:
        msg = f'Page config level must be one of: {allowed}'
        raise ValueError(msg)
    config_name = 'labconfig'
    if level == 'all':
        return ConfigManager(config_dir_name=config_name)
    if level == 'user':
        config_dir = jupyter_config_dir()
    elif level == 'sys_prefix':
        from jupyter_core.paths import ENV_CONFIG_PATH
        config_dir = ENV_CONFIG_PATH[0]
    else:
        config_dir = SYSTEM_CONFIG_PATH[0]
    full_config_path = osp.join(config_dir, config_name)
    return ConfigManager(read_config_path=[full_config_path], write_config_dir=full_config_path)