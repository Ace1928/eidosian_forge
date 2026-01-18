from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import importlib
import inspect
import json
import logging
import os
import time
import threading
import traceback
import asyncio
import sh
import shlex
import shutil
import subprocess
import uuid
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.escape
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.webapp.run_mocks.mock_turk_manager import MockTurkManager
from typing import Dict, Any
from parlai import __path__ as parlai_path  # type: ignore
class AppHandler(tornado.web.RequestHandler):

    def initialize(self, app):
        self.state = app.state
        self.subs = app.subs
        self.sources = app.sources
        self.port = app.port
        self.data_handler = app.data_handler

    def get(self, args):
        self.render('index.html', initial_location=args)
        return