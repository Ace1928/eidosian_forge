from __future__ import annotations
import sys
from typing import Any
from jupyter_server.extension.application import ExtensionApp, ExtensionAppJinjaMixin
from tornado.ioloop import IOLoop
from .handlers import LabConfig, add_handlers
from .process import Process
class ProcessApp(ExtensionAppJinjaMixin, LabConfig, ExtensionApp):
    """A jupyterlab app that runs a separate process and exits on completion."""
    load_other_extensions = True
    open_browser = False

    def get_command(self) -> tuple[list[str], dict[str, Any]]:
        """Get the command and kwargs to run with `Process`.
        This is intended to be overridden.
        """
        return ([sys.executable, '--version'], {})

    def initialize_settings(self) -> None:
        """Start the application."""
        IOLoop.current().add_callback(self._run_command)

    def initialize_handlers(self) -> None:
        """Initialize the handlers."""
        add_handlers(self.handlers, self)

    def _run_command(self) -> None:
        command, kwargs = self.get_command()
        kwargs.setdefault('logger', self.log)
        future = Process(command, **kwargs).wait_async()
        IOLoop.current().add_future(future, self._process_finished)

    def _process_finished(self, future: Any) -> None:
        try:
            IOLoop.current().stop()
            sys.exit(future.result())
        except Exception as e:
            self.log.error(str(e))
            sys.exit(1)