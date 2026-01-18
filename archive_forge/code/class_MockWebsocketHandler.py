import json
import os
import pathlib
import shutil
from pathlib import Path
from typing import Text
from jupyter_server.serverapp import ServerApp
from pytest import fixture
from tornado.httpserver import HTTPRequest
from tornado.httputil import HTTPServerRequest
from tornado.queues import Queue
from tornado.web import Application
from jupyter_lsp import LanguageServerManager
from jupyter_lsp.constants import APP_CONFIG_D_SECTIONS
from jupyter_lsp.handlers import LanguageServersHandler, LanguageServerWebSocketHandler
class MockWebsocketHandler(LanguageServerWebSocketHandler):
    _messages_wrote = None
    _ping_sent = None

    def __init__(self):
        self.request = HTTPServerRequest()
        self.application = Application()

    def initialize(self, manager):
        super().initialize(manager)
        self._messages_wrote = Queue()
        self._ping_sent = False

    def write_message(self, message: Text) -> None:
        self.log.warning('write_message %s', message)
        self._messages_wrote.put_nowait(message)

    def send_ping(self):
        self._ping_sent = True