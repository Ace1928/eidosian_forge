import dataclasses
import json
import os
import sys
from jupyter_core.application import JupyterApp, NoStart, base_aliases, base_flags
from jupyter_server._version import version_info as jpserver_version_info
from jupyter_server.serverapp import flags
from jupyter_server.utils import url_path_join as ujoin
from jupyterlab_server import (
from notebook_shim.shim import NotebookConfigShimMixin
from traitlets import Bool, Instance, Type, Unicode, default
from ._version import __version__
from .commands import (
from .coreconfig import CoreConfig
from .debuglog import DebugLogFileMixin
from .extensions import MANAGERS as EXT_MANAGERS
from .extensions.manager import PluginManager
from .extensions.readonly import ReadOnlyExtensionManager
from .handlers.announcements import (
from .handlers.build_handler import Builder, BuildHandler, build_path
from .handlers.error_handler import ErrorHandler
from .handlers.extension_manager_handler import ExtensionHandler, extensions_handler_path
from .handlers.plugin_manager_handler import PluginHandler, plugins_handler_path
class LabWorkspaceImportApp(WorkspaceImportApp):
    version = version

    @default('workspaces_dir')
    def _default_workspaces_dir(self):
        return get_workspaces_dir()