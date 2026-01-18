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
class RunHandler(BaseHandler):

    def initialize(self, app):
        self.state = app.state
        self.subs = app.subs
        self.sources = app.sources
        self.port = app.port
        self.data_handler = app.data_handler

    def get(self, task_target):
        hits = self.data_handler.get_hits_for_run(task_target)
        processed_hits = []
        for res in hits:
            processed_hits.append(row_to_dict(res))
        assignments = self.data_handler.get_assignments_for_run(task_target)
        pairings = self.data_handler.get_pairings_for_run(task_target)
        processed_assignments = merge_assignments_with_pairings(assignments, pairings, 'task {}'.format(task_target))
        workers = set()
        for assignment in processed_assignments:
            assignment['received_feedback'] = None
            run_id = assignment['run_id']
            conversation_id = assignment['conversation_id']
            worker_id = assignment['worker_id']
            workers.add(worker_id)
            if conversation_id is not None:
                task_data = MTurkDataHandler.get_conversation_data(run_id, conversation_id, worker_id, self.state['is_sandbox'])
                if task_data['data'] is not None:
                    assignment['received_feedback'] = task_data['data'].get('received_feedback')
        worker_data = {}
        for worker in workers:
            worker_data[worker] = row_to_dict(self.data_handler.get_worker_data(worker))
        run_details = row_to_dict(self.data_handler.get_run_data(task_target))
        run_details['run_status'] = 'unimplemented'
        data = {'run_details': run_details, 'worker_details': worker_data, 'assignments': processed_assignments, 'hits': processed_hits}
        self.write(json.dumps(data))