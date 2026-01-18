from __future__ import annotations
import typing as t
from jupyter_core.utils import ensure_async
from jupyter_server._tz import utcnow
from jupyter_server.auth.utils import warn_disabled_authorization
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.base.websocket import WebSocketMixin
from terminado.management import NamedTermManager
from terminado.websocket import TermSocket as BaseTermSocket
from tornado import web
from .base import TerminalsMixin
def _update_activity(self) -> None:
    self.application.settings['terminal_last_activity'] = utcnow()
    if self.term_name in self.terminal_manager.terminals:
        self.terminal_manager.terminals[self.term_name].last_activity = utcnow()