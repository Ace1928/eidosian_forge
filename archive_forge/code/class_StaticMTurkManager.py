import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.mturk_utils as mturk_utils
import parlai.mturk.core.dev.server_utils as server_utils
import parlai.mturk.core.dev.shared_utils as shared_utils
class StaticMTurkManager(MTurkManager):
    """
    Manages interactions between MTurk agents and tasks, the task launching workflow,
    and more, but only for tasks that require just 2 connections to the server: an
    initial task request and the submission of results.
    """

    def __init__(self, opt, is_test=False):
        """
        No interaction means only ever one agent, so that's what we get.
        """
        opt['max_connections'] = 0
        opt['count_complete'] = True
        opt['frontend_template_type'] = 'static'
        super().__init__(opt, ['worker'], is_test, use_db=True)
        self.hit_mult = 1
        self.required_hits = self.num_conversations

    def _assert_opts(self):
        """
        Manages ensuring everything about the passed in options make sense in that they
        don't conflict in some way or another.
        """
        if self.opt.get('allow_reviews'):
            shared_utils.print_and_log(logging.WARN, '[OPT CONFIGURATION ISSUE] allow_reviews is not supported on single person tasks.', should_print=True)
            self.opt['allow_reviews'] = False
        if self.opt.get('frontend_version', 0) < 1:
            shared_utils.print_and_log(logging.WARN, '[OPT CONFIGURATION ISSUE] Static tasks must use the react version of the frontend.', should_print=True)
            raise Exception('Invalid mturk manager options')

    def _onboard_new_agent(self, agent):
        """
        Override onboarding to go straight to the pool for static stasks.
        """
        self._add_agent_to_pool(agent)