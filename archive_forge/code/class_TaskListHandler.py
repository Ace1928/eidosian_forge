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
class TaskListHandler(BaseHandler):

    def initialize(self, app):
        self.state = app.state
        self.subs = app.subs
        self.sources = app.sources
        self.port = app.port
        self.data_handler = app.data_handler

    def get(self):
        results = {t: {'task_name': t, 'dir': v} for t, v in tasks.items()}
        for task_name, directory in tasks.items():
            results[task_name]['internal'] = 'parlai_internal' in directory
            results[task_name]['has_custom'] = False
            results[task_name]['react_frontend'] = False
            try:
                config = get_config_module(task_name)
                frontend_version = config.task_config.get('frontend_version')
                if frontend_version is not None and frontend_version >= 1:
                    results[task_name]['react_frontend'] = True
                if os.path.isfile(os.path.join(directory, 'frontend', 'components', 'custom.jsx')):
                    results[task_name]['has_custom'] = True
            except Exception as e:
                print('Exception {} when loading task details for {}'.format(e, task_name))
                pass
            results[task_name]['active_runs'] = 'unimplemented'
            results[task_name]['all_runs'] = 'unimplemented'
        self.write(json.dumps(list(results.values())))