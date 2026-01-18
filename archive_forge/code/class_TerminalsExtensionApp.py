from __future__ import annotations
import os
import shlex
import sys
import typing as t
from shutil import which
from jupyter_core.utils import ensure_async
from jupyter_server.extension.application import ExtensionApp
from jupyter_server.transutils import trans
from traitlets import Type
from . import api_handlers, handlers
from .terminalmanager import TerminalManager
class TerminalsExtensionApp(ExtensionApp):
    """A terminals extension app."""
    name = 'jupyter_server_terminals'
    terminal_manager_class: type[TerminalManager] = Type(default_value=TerminalManager, help='The terminal manager class to use.').tag(config=True)
    terminals_available = False

    def initialize_settings(self) -> None:
        """Initialize settings."""
        if not self.serverapp or not self.serverapp.terminals_enabled:
            self.settings.update({'terminals_available': False})
            return
        self.initialize_configurables()
        self.settings.update({'terminals_available': True, 'terminal_manager': self.terminal_manager})

    def initialize_configurables(self) -> None:
        """Initialize configurables."""
        default_shell = 'powershell.exe' if os.name == 'nt' else which('sh')
        assert self.serverapp is not None
        shell_override = self.serverapp.terminado_settings.get('shell_command')
        if isinstance(shell_override, str):
            shell_override = shlex.split(shell_override)
        shell = [os.environ.get('SHELL') or default_shell] if shell_override is None else shell_override
        if os.name != 'nt' and shell_override is None and (not sys.stdout.isatty()):
            shell.append('-l')
        self.terminal_manager = self.terminal_manager_class(shell_command=shell, extra_env={'JUPYTER_SERVER_ROOT': self.serverapp.root_dir, 'JUPYTER_SERVER_URL': self.serverapp.connection_url}, parent=self.serverapp)
        self.terminal_manager.log = self.serverapp.log

    def initialize_handlers(self) -> None:
        """Initialize handlers."""
        if not self.serverapp:
            return
        if not self.serverapp.terminals_enabled:
            self.serverapp.web_app.settings['terminals_available'] = self.settings['terminals_available']
            return
        self.handlers.append(('/terminals/websocket/(\\w+)', handlers.TermSocket, {'term_manager': self.terminal_manager}))
        self.handlers.extend(api_handlers.default_handlers)
        assert self.serverapp is not None
        self.serverapp.web_app.settings['terminal_manager'] = self.terminal_manager
        self.serverapp.web_app.settings['terminals_available'] = self.settings['terminals_available']

    def current_activity(self) -> dict[str, t.Any] | None:
        """Get current activity info."""
        if self.terminals_available:
            terminals = self.terminal_manager.terminals
            if terminals:
                return terminals
        return None

    async def cleanup_terminals(self) -> None:
        """Shutdown all terminals.

        The terminals will shutdown themselves when this process no longer exists,
        but explicit shutdown allows the TerminalManager to cleanup.
        """
        if not self.terminals_available:
            return
        terminal_manager = self.terminal_manager
        n_terminals = len(terminal_manager.list())
        terminal_msg = trans.ngettext('Shutting down %d terminal', 'Shutting down %d terminals', n_terminals)
        self.log.info('%s %% %s', terminal_msg, n_terminals)
        await ensure_async(terminal_manager.terminate_all())

    async def stop_extension(self) -> None:
        """Stop the extension."""
        await self.cleanup_terminals()