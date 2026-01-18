from __future__ import annotations
import importlib
from itertools import starmap
from tornado.gen import multi
from traitlets import Any, Bool, Dict, HasTraits, Instance, List, Unicode, default, observe
from traitlets import validate as validate_trait
from traitlets.config import LoggingConfigurable
from .config import ExtensionConfigManager
from .utils import ExtensionMetadataError, ExtensionModuleNotFound, get_loader, get_metadata
def _load_config_manager(self, config_manager):
    """Actually load our config manager"""
    jpserver_extensions = config_manager.get_jpserver_extensions()
    self.from_jpserver_extensions(jpserver_extensions)