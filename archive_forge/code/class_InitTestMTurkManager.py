import unittest
import os
import time
import json
import threading
import pickle
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.worlds import MTurkOnboardWorld
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.mturk.core.dev.socket_manager import SocketManager, Packet
from parlai.core.params import ParlaiParser
from websocket_server import WebsocketServer
import parlai.mturk.core.dev.mturk_manager as MTurkManagerFile
import parlai.mturk.core.dev.data_model as data_model
class InitTestMTurkManager(unittest.TestCase):
    """
    Unit tests for MTurkManager setup.
    """

    def setUp(self):
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args()
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        self.mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
        self.mturk_manager = MTurkManager(opt=self.opt, mturk_agent_ids=self.mturk_agent_ids, is_test=True)

    def tearDown(self):
        self.mturk_manager.shutdown()

    def test_init(self):
        manager = self.mturk_manager
        opt = self.opt
        self.assertIsNone(manager.server_url)
        self.assertIsNone(manager.topic_arn)
        self.assertIsNone(manager.server_task_name)
        self.assertIsNone(manager.task_group_id)
        self.assertIsNone(manager.run_id)
        self.assertIsNone(manager.task_files_to_copy)
        self.assertIsNone(manager.get_onboard_world)
        self.assertIsNone(manager.socket_manager)
        self.assertFalse(manager.is_shutdown.is_set())
        self.assertFalse(manager.is_unique)
        self.assertEqual(manager.opt, opt)
        self.assertEqual(manager.mturk_agent_ids, self.mturk_agent_ids)
        self.assertEqual(manager.is_sandbox, opt['is_sandbox'])
        self.assertEqual(manager.num_conversations, opt['num_conversations'])
        self.assertEqual(manager.is_sandbox, opt['is_sandbox'])
        self.assertGreaterEqual(manager.required_hits, manager.num_conversations * len(self.mturk_agent_ids))
        self.assertIsNotNone(manager.agent_pool_change_condition)
        self.assertEqual(manager.minimum_messages, opt.get('min_messages', 0))
        self.assertEqual(manager.auto_approve_delay, opt.get('auto_approve_delay', 4 * 7 * 24 * 3600))
        self.assertEqual(manager.has_time_limit, opt.get('max_time', 0) > 0)
        self.assertIsInstance(manager.worker_manager, WorkerManager)
        self.assertEqual(manager.task_state, manager.STATE_CREATED)

    def test_init_state(self):
        manager = self.mturk_manager
        manager._init_state()
        self.assertEqual(manager.agent_pool, [])
        self.assertEqual(manager.hit_id_list, [])
        self.assertEqual(manager.conversation_index, 0)
        self.assertEqual(manager.started_conversations, 0)
        self.assertEqual(manager.completed_conversations, 0)
        self.assertEqual(manager.task_threads, [])
        self.assertTrue(manager.accepting_workers, True)
        self.assertIsNone(manager.qualifications)
        self.assertGreater(manager.time_limit_checked, time.time() - 1)
        self.assertEqual(manager.task_state, manager.STATE_INIT_RUN)