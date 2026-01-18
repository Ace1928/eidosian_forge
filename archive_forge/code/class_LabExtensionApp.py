import os
import sys
from copy import copy
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Bool, Instance, List, Unicode, default
from jupyterlab.coreconfig import CoreConfig
from jupyterlab.debuglog import DebugLogFileMixin
from .commands import (
from .federated_labextensions import build_labextension, develop_labextension_py, watch_labextension
from .labapp import LabApp
class LabExtensionApp(JupyterApp):
    """Base jupyter labextension command entry point"""
    name = 'jupyter labextension'
    version = VERSION
    description = 'Work with JupyterLab extensions'
    examples = _EXAMPLES
    subcommands = {'install': (InstallLabExtensionApp, 'Install labextension(s)'), 'update': (UpdateLabExtensionApp, 'Update labextension(s)'), 'uninstall': (UninstallLabExtensionApp, 'Uninstall labextension(s)'), 'list': (ListLabExtensionsApp, 'List labextensions'), 'link': (LinkLabExtensionApp, 'Link labextension(s)'), 'unlink': (UnlinkLabExtensionApp, 'Unlink labextension(s)'), 'enable': (EnableLabExtensionsApp, 'Enable labextension(s)'), 'disable': (DisableLabExtensionsApp, 'Disable labextension(s)'), 'lock': (LockLabExtensionsApp, 'Lock labextension(s)'), 'unlock': (UnlockLabExtensionsApp, 'Unlock labextension(s)'), 'check': (CheckLabExtensionsApp, 'Check labextension(s)'), 'develop': (DevelopLabExtensionApp, '(developer) Develop labextension(s)'), 'build': (BuildLabExtensionApp, '(developer) Build labextension'), 'watch': (WatchLabExtensionApp, '(developer) Watch labextension')}

    def start(self):
        """Perform the App's functions as configured"""
        super().start()
        subcmds = ', '.join(sorted(self.subcommands))
        self.exit('Please supply at least one subcommand: %s' % subcmds)