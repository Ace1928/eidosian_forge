import logging
import os
import sys
import warnings
from traitlets.config.loader import Config
from traitlets.config.application import boolean_flag, catch_config_error
from IPython.core import release
from IPython.core import usage
from IPython.core.completer import IPCompleter
from IPython.core.crashhandler import CrashHandler
from IPython.core.formatters import PlainTextFormatter
from IPython.core.history import HistoryManager
from IPython.core.application import (
from IPython.core.magic import MagicsManager
from IPython.core.magics import (
from IPython.core.shellapp import (
from IPython.extensions.storemagic import StoreMagics
from .interactiveshell import TerminalInteractiveShell
from IPython.paths import get_ipython_dir
from traitlets import (
class TerminalIPythonApp(BaseIPythonApplication, InteractiveShellApp):
    name = 'ipython'
    description = usage.cl_usage
    crash_handler_class = IPAppCrashHandler
    examples = _examples
    flags = flags
    aliases = aliases
    classes = List()
    interactive_shell_class = Type(klass=object, default_value=TerminalInteractiveShell, help='Class to use to instantiate the TerminalInteractiveShell object. Useful for custom Frontends').tag(config=True)

    @default('classes')
    def _classes_default(self):
        """This has to be in a method, for TerminalIPythonApp to be available."""
        return [InteractiveShellApp, self.__class__, TerminalInteractiveShell, HistoryManager, MagicsManager, ProfileDir, PlainTextFormatter, IPCompleter, ScriptMagics, LoggingMagics, StoreMagics]
    subcommands = dict(profile=('IPython.core.profileapp.ProfileApp', 'Create and manage IPython profiles.'), kernel=('ipykernel.kernelapp.IPKernelApp', 'Start a kernel without an attached frontend.'), locate=('IPython.terminal.ipapp.LocateIPythonApp', LocateIPythonApp.description), history=('IPython.core.historyapp.HistoryApp', 'Manage the IPython history database.'))
    auto_create = Bool(True).tag(config=True)
    quick = Bool(False, help='Start IPython quickly by skipping the loading of config files.').tag(config=True)

    @observe('quick')
    def _quick_changed(self, change):
        if change['new']:
            self.load_config_file = lambda *a, **kw: None
    display_banner = Bool(True, help='Whether to display a banner upon starting IPython.').tag(config=True)
    force_interact = Bool(False, help="If a command or file is given via the command-line,\n        e.g. 'ipython foo.py', start an interactive shell after executing the\n        file or command.").tag(config=True)

    @observe('force_interact')
    def _force_interact_changed(self, change):
        if change['new']:
            self.interact = True

    @observe('file_to_run', 'code_to_run', 'module_to_run')
    def _file_to_run_changed(self, change):
        new = change['new']
        if new:
            self.something_to_run = True
        if new and (not self.force_interact):
            self.interact = False
    something_to_run = Bool(False)

    @catch_config_error
    def initialize(self, argv=None):
        """Do actions after construct, but before starting the app."""
        super(TerminalIPythonApp, self).initialize(argv)
        if self.subapp is not None:
            return
        if self.extra_args and (not self.something_to_run):
            self.file_to_run = self.extra_args[0]
        self.init_path()
        self.init_shell()
        self.init_banner()
        self.init_gui_pylab()
        self.init_extensions()
        self.init_code()

    def init_shell(self):
        """initialize the InteractiveShell instance"""
        self.shell = self.interactive_shell_class.instance(parent=self, profile_dir=self.profile_dir, ipython_dir=self.ipython_dir, user_ns=self.user_ns)
        self.shell.configurables.append(self)

    def init_banner(self):
        """optionally display the banner"""
        if self.display_banner and self.interact:
            self.shell.show_banner()
        if self.log_level <= logging.INFO:
            print()

    def _pylab_changed(self, name, old, new):
        """Replace --pylab='inline' with --pylab='auto'"""
        if new == 'inline':
            warnings.warn("'inline' not available as pylab backend, using 'auto' instead.")
            self.pylab = 'auto'

    def start(self):
        if self.subapp is not None:
            return self.subapp.start()
        if self.interact:
            self.log.debug("Starting IPython's mainloop...")
            self.shell.mainloop()
        else:
            self.log.debug('IPython not interactive...')
            self.shell.restore_term_title()
            if not self.shell.last_execution_succeeded:
                sys.exit(1)