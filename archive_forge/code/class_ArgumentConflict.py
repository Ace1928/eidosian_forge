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
class ArgumentConflict(ValueError):
    pass