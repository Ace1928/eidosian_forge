import asyncio
import inspect
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from os import path as osp
from jupyter_server.serverapp import aliases, flags
from jupyter_server.utils import pathname2url, urljoin
from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError
from tornado.websocket import WebSocketClosedError
from traitlets import Bool, Unicode
from .labapp import LabApp, get_app_dir
from .tests.test_app import TestEnv
class BrowserApp(LabApp):
    """An app the launches JupyterLab and waits for it to start up, checking for
    JS console errors, JS errors, and Python logged errors.
    """
    name = __name__
    open_browser = False
    serverapp_config = {'base_url': '/foo/'}
    default_url = Unicode('/lab?reset', config=True, help='The default URL to redirect to from `/`')
    ip = '127.0.0.1'
    flags = test_flags
    aliases = test_aliases
    test_browser = Bool(True)

    def initialize_settings(self):
        self.settings.setdefault('page_config_data', {})
        self.settings['page_config_data']['browserTest'] = True
        self.settings['page_config_data']['buildAvailable'] = False
        self.settings['page_config_data']['exposeAppInBrowser'] = True
        super().initialize_settings()

    def initialize_handlers(self):
        func = run_browser if self.test_browser else lambda url: 0
        if os.name == 'nt' and func == run_browser:
            func = run_browser_sync
        run_test(self.serverapp, func)
        super().initialize_handlers()