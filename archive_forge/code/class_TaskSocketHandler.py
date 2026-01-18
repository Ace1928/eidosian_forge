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
class TaskSocketHandler(tornado.websocket.WebSocketHandler):

    def initialize(self, app):
        self.app = app
        self.sources = app.sources

    def check_origin(self, origin):
        return True

    def _run_socket(self):
        time.sleep(2)
        asyncio.set_event_loop(asyncio.new_event_loop())
        while self.alive and self.app.task_manager is not None:
            try:
                self.write_message(json.dumps({'data': [agent.get_update_packet() for agent in self.app.task_manager.agents], 'command': 'sync'}))
                time.sleep(0.2)
            except tornado.websocket.WebSocketClosedError:
                self.alive = False
                self.app.task_manager.timeout_all_agents()

    def open(self):
        self.sid = str(hex(int(time.time() * 10000000))[2:])
        self.alive = True
        if self not in list(self.sources.values()):
            self.sources[self.sid] = self
        logging.info('Opened task socket from ip: {}'.format(self.request.remote_ip))
        self.write_message(json.dumps({'command': 'alive', 'data': 'socket_alive'}))
        t = threading.Thread(target=self._run_socket)
        t.start()

    def on_message(self, message):
        logging.info('from frontend client: {}'.format(message))
        msg = tornado.escape.json_decode(tornado.escape.to_basestring(message))
        message = msg['text']
        task_data = msg['task_data']
        sender_id = msg['sender']
        agent_id = msg['id']
        act = {'id': agent_id, 'task_data': task_data, 'text': message, 'message_id': str(uuid.uuid4())}
        t = threading.Thread(target=self.app.task_manager.on_new_message, args=(sender_id, PacketWrap(act)), daemon=True)
        t.start()

    def on_close(self):
        self.alive = False
        if self in list(self.sources.values()):
            self.sources.pop(self.sid, None)