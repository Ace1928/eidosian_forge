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
class TestMTurkManagerUnitFunctions(unittest.TestCase):
    """
    Tests some of the simpler MTurkManager functions that don't require much additional
    state to run.
    """

    def setUp(self):
        self.fake_socket = MockSocket()
        time.sleep(0.1)
        argparser = ParlaiParser(False, False)
        argparser.add_parlai_data_path()
        argparser.add_mturk_args()
        self.opt = argparser.parse_args()
        self.opt['task'] = 'unittest'
        self.opt['assignment_duration_in_seconds'] = 6
        self.mturk_agent_ids = ['mturk_agent_1', 'mturk_agent_2']
        self.mturk_manager = MTurkManager(opt=self.opt, mturk_agent_ids=self.mturk_agent_ids, is_test=True)
        self.mturk_manager._init_state()
        self.mturk_manager.send_state_change = mock.MagicMock()
        self.mturk_manager.port = self.fake_socket.port
        self.agent_1 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_1, TEST_ASSIGNMENT_ID_1, TEST_WORKER_ID_1)
        self.agent_2 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_2, TEST_ASSIGNMENT_ID_2, TEST_WORKER_ID_2)
        self.agent_3 = MTurkAgent(self.opt, self.mturk_manager, TEST_HIT_ID_3, TEST_ASSIGNMENT_ID_3, TEST_WORKER_ID_3)

    def tearDown(self):
        self.mturk_manager.shutdown()
        self.fake_socket.close()

    def test_move_to_waiting(self):
        manager = self.mturk_manager
        manager.socket_manager = mock.MagicMock()
        manager.socket_manager.close_channel = mock.MagicMock()
        manager.force_expire_hit = mock.MagicMock()
        self.agent_1.set_status(AssignState.STATUS_DISCONNECT)
        self.agent_1.reduce_state = mock.MagicMock()
        self.agent_2.reduce_state = mock.MagicMock()
        self.agent_3.reduce_state = mock.MagicMock()
        manager._move_agents_to_waiting([self.agent_1])
        self.agent_1.reduce_state.assert_called_once()
        manager.socket_manager.close_channel.assert_called_once_with(self.agent_1.get_connection_id())
        manager.force_expire_hit.assert_not_called()
        manager.socket_manager.close_channel.reset_mock()
        manager._move_agents_to_waiting([self.agent_2])
        self.agent_2.reduce_state.assert_not_called()
        manager.socket_manager.close_channel.assert_not_called()
        manager.force_expire_hit.assert_not_called()
        manager.accepting_workers = False
        manager._move_agents_to_waiting([self.agent_3])
        self.agent_3.reduce_state.assert_not_called()
        manager.socket_manager.close_channel.assert_not_called()
        manager.force_expire_hit.assert_called_once_with(self.agent_3.worker_id, self.agent_3.assignment_id)

    def test_socket_setup(self):
        """
        Basic socket setup should fail when not in correct state, but succeed otherwise.
        """
        self.mturk_manager.task_state = self.mturk_manager.STATE_CREATED
        with self.assertRaises(AssertionError):
            self.mturk_manager._setup_socket()
        self.mturk_manager.task_group_id = 'TEST_GROUP_ID'
        self.mturk_manager.server_url = 'https://127.0.0.1'
        self.mturk_manager.task_state = self.mturk_manager.STATE_INIT_RUN
        self.mturk_manager._setup_socket()
        self.assertIsInstance(self.mturk_manager.socket_manager, SocketManager)

    def test_worker_alive(self):
        manager = self.mturk_manager
        manager.task_group_id = 'TEST_GROUP_ID'
        manager.server_url = 'https://127.0.0.1'
        manager.task_state = manager.STATE_ACCEPTING_WORKERS
        manager._setup_socket()
        manager.force_expire_hit = mock.MagicMock()
        manager._onboard_new_agent = mock.MagicMock()
        manager.socket_manager.open_channel = mock.MagicMock(wraps=manager.socket_manager.open_channel)
        manager.worker_manager.worker_alive = mock.MagicMock(wraps=manager.worker_manager.worker_alive)
        open_channel = manager.socket_manager.open_channel
        worker_alive = manager.worker_manager.worker_alive
        alive_packet = Packet('', '', '', '', '', {'worker_id': TEST_WORKER_ID_1, 'hit_id': TEST_HIT_ID_1, 'assignment_id': None, 'conversation_id': None}, '')
        manager._on_alive(alive_packet)
        open_channel.assert_not_called()
        worker_alive.assert_not_called()
        manager._onboard_new_agent.assert_not_called()
        alive_packet = Packet('', '', '', '', '', {'worker_id': TEST_WORKER_ID_1, 'hit_id': TEST_HIT_ID_1, 'assignment_id': TEST_ASSIGNMENT_ID_1, 'conversation_id': None}, '')
        manager.accepting_workers = False
        manager._on_alive(alive_packet)
        open_channel.assert_called_once_with(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        worker_alive.assert_called_once_with(TEST_WORKER_ID_1)
        worker_state = manager.worker_manager._get_worker(TEST_WORKER_ID_1)
        self.assertIsNotNone(worker_state)
        open_channel.reset_mock()
        worker_alive.reset_mock()
        manager.force_expire_hit.assert_called_once_with(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        manager._onboard_new_agent.assert_not_called()
        manager.force_expire_hit.reset_mock()
        manager.accepting_workers = True
        manager._on_alive(alive_packet)
        open_channel.assert_called_once_with(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        worker_alive.assert_called_once_with(TEST_WORKER_ID_1)
        manager._onboard_new_agent.assert_called_once()
        manager._onboard_new_agent.reset_mock()
        manager.force_expire_hit.assert_not_called()
        agent = manager.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_1)
        self.assertIsInstance(agent, MTurkAgent)
        self.assertEqual(agent.get_status(), AssignState.STATUS_NONE)
        agent.set_status(AssignState.STATUS_IN_TASK)
        alive_packet = Packet('', '', '', '', '', {'worker_id': TEST_WORKER_ID_1, 'hit_id': TEST_HIT_ID_1, 'assignment_id': TEST_ASSIGNMENT_ID_2, 'conversation_id': None}, '')
        manager.opt['allowed_conversations'] = 1
        manager._on_alive(alive_packet)
        manager.force_expire_hit.assert_called_once()
        manager._onboard_new_agent.assert_not_called()
        manager.force_expire_hit.reset_mock()
        agent.set_status(AssignState.STATUS_DONE)
        alive_packet = Packet('', '', '', '', '', {'worker_id': TEST_WORKER_ID_1, 'hit_id': TEST_HIT_ID_1, 'assignment_id': TEST_ASSIGNMENT_ID_2, 'conversation_id': None}, '')
        manager.is_unique = True
        manager._on_alive(alive_packet)
        manager.force_expire_hit.assert_called_once()
        manager._onboard_new_agent.assert_not_called()
        manager.force_expire_hit.reset_mock()

    def test_mturk_messages(self):
        """
        Ensure incoming messages work as expected.
        """
        manager = self.mturk_manager
        manager.task_group_id = 'TEST_GROUP_ID'
        manager.server_url = 'https://127.0.0.1'
        manager.task_state = manager.STATE_ACCEPTING_WORKERS
        manager._setup_socket()
        manager.force_expire_hit = mock.MagicMock()
        manager._on_socket_dead = mock.MagicMock()
        alive_packet = Packet('', '', '', '', '', {'worker_id': TEST_WORKER_ID_1, 'hit_id': TEST_HIT_ID_1, 'assignment_id': TEST_ASSIGNMENT_ID_1, 'conversation_id': None}, '')
        manager._on_alive(alive_packet)
        agent = manager.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_1)
        self.assertIn(agent.get_status(), [AssignState.STATUS_NONE, AssignState.STATUS_WAITING])
        self.assertIsInstance(agent, MTurkAgent)
        manager._on_socket_dead = mock.MagicMock()
        message_packet = Packet('', '', '', '', TEST_ASSIGNMENT_ID_1, {'text': MTurkManagerFile.SNS_ASSIGN_ABANDONDED}, '')
        manager._handle_mturk_message(message_packet)
        manager._on_socket_dead.assert_called_once_with(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        manager._on_socket_dead.reset_mock()
        message_packet = Packet('', '', '', '', TEST_ASSIGNMENT_ID_1, {'text': MTurkManagerFile.SNS_ASSIGN_RETURNED}, '')
        agent.hit_is_returned = False
        manager._handle_mturk_message(message_packet)
        manager._on_socket_dead.assert_called_once_with(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        manager._on_socket_dead.reset_mock()
        self.assertTrue(agent.hit_is_returned)
        message_packet = Packet('', '', '', '', TEST_ASSIGNMENT_ID_1, {'text': MTurkManagerFile.SNS_ASSIGN_SUBMITTED}, '')
        agent.hit_is_complete = False
        manager._handle_mturk_message(message_packet)
        manager._on_socket_dead.assert_not_called()
        self.assertTrue(agent.hit_is_complete)

    def test_new_message(self):
        """
        test on_new_message.
        """
        alive_packet = Packet('', TEST_WORKER_ID_1, '', '', '', {'worker_id': TEST_WORKER_ID_1, 'hit_id': TEST_HIT_ID_1, 'assignment_id': TEST_ASSIGNMENT_ID_1, 'conversation_id': None}, '')
        message_packet = Packet('', '', MTurkManagerFile.AMAZON_SNS_NAME, '', TEST_ASSIGNMENT_ID_1, {'text': MTurkManagerFile.SNS_ASSIGN_SUBMITTED}, '')
        manager = self.mturk_manager
        manager._handle_mturk_message = mock.MagicMock()
        manager.worker_manager.route_packet = mock.MagicMock()
        manager._on_new_message(alive_packet)
        manager._handle_mturk_message.assert_not_called()
        manager.worker_manager.route_packet.assert_called_once_with(alive_packet)
        manager.worker_manager.route_packet.reset_mock()
        manager._on_new_message(message_packet)
        manager._handle_mturk_message.assert_called_once_with(message_packet)
        manager.worker_manager.route_packet.assert_not_called()

    def test_onboarding_function(self):
        manager = self.mturk_manager
        manager.get_onboard_world = mock.MagicMock(wraps=get_onboard_world)
        manager.send_message = mock.MagicMock()
        manager._move_agents_to_waiting = mock.MagicMock()
        manager.worker_manager.get_agent_for_assignment = mock.MagicMock(return_value=self.agent_1)
        onboard_threads = manager.assignment_to_onboard_thread
        did_launch = manager._onboard_new_agent(self.agent_1)
        assert_equal_by(onboard_threads[self.agent_1.assignment_id].isAlive, True, 0.2)
        time.sleep(0.1)
        self.assertTrue(did_launch)
        manager.get_onboard_world.assert_called_with(self.agent_1)
        manager.get_onboard_world.reset_mock()
        did_launch = manager._onboard_new_agent(self.agent_1)
        manager.worker_manager.get_agent_for_assignment.assert_not_called()
        manager.get_onboard_world.assert_not_called()
        self.assertFalse(did_launch)
        assert_equal_by(onboard_threads[self.agent_1.assignment_id].isAlive, False, 3)
        manager._move_agents_to_waiting.assert_called_once()
        did_launch = manager._onboard_new_agent(self.agent_1)
        self.assertFalse(did_launch)
        self.agent_1.set_status(AssignState.STATUS_NONE)
        did_launch = manager._onboard_new_agent(self.agent_1)
        self.assertTrue(did_launch)

    def test_agents_incomplete(self):
        agents = [self.agent_1, self.agent_2, self.agent_3]
        manager = self.mturk_manager
        manager.send_state_change = mock.MagicMock()
        self.assertFalse(manager._no_agents_incomplete(agents))
        self.agent_1.set_status(AssignState.STATUS_DISCONNECT)
        self.assertFalse(manager._no_agents_incomplete(agents))
        self.agent_2.set_status(AssignState.STATUS_DONE)
        self.assertFalse(manager._no_agents_incomplete(agents))
        self.agent_3.set_status(AssignState.STATUS_PARTNER_DISCONNECT)
        self.assertFalse(manager._no_agents_incomplete(agents))
        self.agent_1.set_status(AssignState.STATUS_DONE)
        self.assertFalse(manager._no_agents_incomplete(agents))
        self.agent_3.set_status(AssignState.STATUS_DONE)
        self.assertTrue(manager._no_agents_incomplete(agents))

    def test_world_types(self):
        onboard_type = 'o_12345'
        waiting_type = 'w_12345'
        task_type = 't_12345'
        garbage_type = 'g_12345'
        manager = self.mturk_manager
        self.assertTrue(manager.is_onboarding_world(onboard_type))
        self.assertTrue(manager.is_task_world(task_type))
        self.assertTrue(manager.is_waiting_world(waiting_type))
        for world_type in [waiting_type, task_type, garbage_type]:
            self.assertFalse(manager.is_onboarding_world(world_type))
        for world_type in [onboard_type, task_type, garbage_type]:
            self.assertFalse(manager.is_waiting_world(world_type))
        for world_type in [waiting_type, onboard_type, garbage_type]:
            self.assertFalse(manager.is_task_world(world_type))

    def test_turk_timeout(self):
        """
        Timeout should send expiration message to worker and be treated as a disconnect
        event.
        """
        manager = self.mturk_manager
        manager.force_expire_hit = mock.MagicMock()
        manager._handle_agent_disconnect = mock.MagicMock()
        manager.handle_turker_timeout(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)
        manager.force_expire_hit.assert_called_once()
        call_args = manager.force_expire_hit.call_args
        self.assertEqual(call_args[0][0], TEST_WORKER_ID_1)
        self.assertEqual(call_args[0][1], TEST_ASSIGNMENT_ID_1)
        manager._handle_agent_disconnect.assert_called_once_with(TEST_WORKER_ID_1, TEST_ASSIGNMENT_ID_1)

    def test_wait_for_task_expirations(self):
        """
        Ensure waiting for expiration time actually works out.
        """
        manager = self.mturk_manager
        manager.opt['assignment_duration_in_seconds'] = 0.5
        manager.expire_all_unassigned_hits = mock.MagicMock()
        manager.update_hit_status = mock.MagicMock()
        manager.hit_id_list = [1, 2, 3]

        def run_task_wait():
            manager._wait_for_task_expirations()
        wait_thread = threading.Thread(target=run_task_wait, daemon=True)
        wait_thread.start()
        time.sleep(0.1)
        self.assertTrue(wait_thread.isAlive())
        assert_equal_by(wait_thread.isAlive, False, 3)

    def test_mark_workers_done(self):
        manager = self.mturk_manager
        manager.send_state_change = mock.MagicMock()
        manager.give_worker_qualification = mock.MagicMock()
        manager._log_working_time = mock.MagicMock()
        manager.has_time_limit = False
        self.agent_1.set_status(AssignState.STATUS_DISCONNECT)
        manager.mark_workers_done([self.agent_1])
        self.assertEqual(AssignState.STATUS_DISCONNECT, self.agent_1.get_status())
        manager.is_unique = True
        with self.assertRaises(AssertionError):
            manager.mark_workers_done([self.agent_2])
        manager.give_worker_qualification.assert_not_called()
        manager.unique_qual_name = 'fake_qual_name'
        manager.mark_workers_done([self.agent_2])
        manager.give_worker_qualification.assert_called_once_with(self.agent_2.worker_id, 'fake_qual_name')
        self.assertEqual(self.agent_2.get_status(), AssignState.STATUS_DONE)
        manager.is_unique = False
        manager.has_time_limit = True
        manager.mark_workers_done([self.agent_3])
        self.assertEqual(self.agent_3.get_status(), AssignState.STATUS_DONE)
        manager._log_working_time.assert_called_once_with(self.agent_3)