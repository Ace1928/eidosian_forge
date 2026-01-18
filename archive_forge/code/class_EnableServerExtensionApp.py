from __future__ import annotations
import logging
import os
import sys
import typing as t
from jupyter_core.application import JupyterApp
from jupyter_core.paths import ENV_CONFIG_PATH, SYSTEM_CONFIG_PATH, jupyter_config_dir
from tornado.log import LogFormatter
from traitlets import Bool
from jupyter_server._version import __version__
from jupyter_server.extension.config import ExtensionConfigManager
from jupyter_server.extension.manager import ExtensionManager, ExtensionPackage
class EnableServerExtensionApp(ToggleServerExtensionApp):
    """An App that enables (and validates) Server Extensions"""
    name = 'jupyter server extension enable'
    description = '\n    Enable a server extension in configuration.\n\n    Usage\n        jupyter server extension enable [--system|--sys-prefix]\n    '
    _toggle_value = True
    _toggle_pre_message = 'enabling'
    _toggle_post_message = 'enabled'