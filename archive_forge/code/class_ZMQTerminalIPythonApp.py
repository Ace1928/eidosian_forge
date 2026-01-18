from __future__ import print_function
import signal
import sys
from traitlets import (
from traitlets.config import catch_config_error, boolean_flag
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from jupyter_client.consoleapp import (
from jupyter_console.ptshell import ZMQTerminalInteractiveShell
from jupyter_console import __version__
class ZMQTerminalIPythonApp(JupyterApp, JupyterConsoleApp):
    name = 'jupyter-console'
    version = __version__
    'Start a terminal frontend to the IPython zmq kernel.'
    description = '\n        The Jupyter terminal-based Console.\n\n        This launches a Console application inside a terminal.\n\n        The Console supports various extra features beyond the traditional\n        single-process Terminal IPython shell, such as connecting to an\n        existing ipython session, via:\n\n            jupyter console --existing\n\n        where the previous session could have been created by another ipython\n        console, an ipython qtconsole, or by opening an ipython notebook.\n\n    '
    examples = _examples
    classes = [ZMQTerminalInteractiveShell] + JupyterConsoleApp.classes
    flags = Dict(flags)
    aliases = Dict(aliases)
    frontend_aliases = Any(frontend_aliases)
    frontend_flags = Any(frontend_flags)
    subcommands = Dict()
    force_interact = True

    def parse_command_line(self, argv=None):
        super(ZMQTerminalIPythonApp, self).parse_command_line(argv)
        self.build_kernel_argv(self.extra_args)

    def init_shell(self):
        JupyterConsoleApp.initialize(self)
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.shell = ZMQTerminalInteractiveShell.instance(parent=self, manager=self.kernel_manager, client=self.kernel_client, confirm_exit=self.confirm_exit)
        self.shell.own_kernel = not self.existing

    def init_gui_pylab(self):
        pass

    def handle_sigint(self, *args):
        if self.shell._executing:
            if self.kernel_manager:
                self.kernel_manager.interrupt_kernel()
            else:
                print("ERROR: Cannot interrupt kernels we didn't start.", file=sys.stderr)
        else:
            raise KeyboardInterrupt

    @catch_config_error
    def initialize(self, argv=None):
        """Do actions after construct, but before starting the app."""
        super(ZMQTerminalIPythonApp, self).initialize(argv)
        if self._dispatching:
            return
        self.init_shell()
        self.init_banner()

    def init_banner(self):
        """optionally display the banner"""
        self.shell.show_banner()

    def start(self):
        super(ZMQTerminalIPythonApp, self).start()
        self.log.debug('Starting the jupyter console mainloop...')
        self.shell.mainloop()