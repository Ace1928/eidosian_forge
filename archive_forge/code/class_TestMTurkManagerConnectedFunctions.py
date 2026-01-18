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
class TestMTurkManagerConnectedFunctions(unittest.TestCase):
    """
    Semi-unit semi-integration tests on the more state-dependent MTurkManager
    functionality.
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
        self.mturk_manager.port = self.fake_socket.port
        self.mturk_manager._onboard_new_agent = mock.MagicMock()
        self.mturk_manager._wait_for_task_expirations = mock.MagicMock()
        self.mturk_manager.task_group_id = 'TEST_GROUP_ID'
        self.mturk_manager.server_url = 'https://127.0.0.1'
        self.mturk_manager.task_state = self.mturk_manager.STATE_ACCEPTING_WORKERS
        self.mturk_manager._setup_socket()
        alive_packet = Packet('', '', '', '', '', {'worker_id': TEST_WORKER_ID_1, 'hit_id': TEST_HIT_ID_1, 'assignment_id': TEST_ASSIGNMENT_ID_1, 'conversation_id': None}, '')
        self.mturk_manager._on_alive(alive_packet)
        alive_packet = Packet('', '', '', '', '', {'worker_id': TEST_WORKER_ID_2, 'hit_id': TEST_HIT_ID_2, 'assignment_id': TEST_ASSIGNMENT_ID_2, 'conversation_id': None}, '')
        self.mturk_manager._on_alive(alive_packet)
        self.agent_1 = self.mturk_manager.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_1)
        self.agent_2 = self.mturk_manager.worker_manager.get_agent_for_assignment(TEST_ASSIGNMENT_ID_2)

    def tearDown(self):
        self.mturk_manager.shutdown()
        self.fake_socket.close()

    def test_socket_dead(self):
        """
        Test all states of socket dead calls.
        """
        manager = self.mturk_manager
        agent = self.agent_1
        worker_id = agent.worker_id
        assignment_id = agent.assignment_id
        manager.socket_manager.close_channel = mock.MagicMock()
        agent.reduce_state = mock.MagicMock()
        agent.set_status = mock.MagicMock(wraps=agent.set_status)
        manager._handle_agent_disconnect = mock.MagicMock(wraps=manager._handle_agent_disconnect)
        agent.set_status(AssignState.STATUS_NONE)
        agent.set_status.reset_mock()
        manager._on_socket_dead(worker_id, assignment_id)
        self.assertEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
        agent.reduce_state.assert_called_once()
        manager.socket_manager.close_channel.assert_called_once_with(agent.get_connection_id())
        manager._handle_agent_disconnect.assert_not_called()
        agent.set_status(AssignState.STATUS_ONBOARDING)
        agent.set_status.reset_mock()
        agent.reduce_state.reset_mock()
        manager.socket_manager.close_channel.reset_mock()
        self.assertFalse(agent.disconnected)
        manager._on_socket_dead(worker_id, assignment_id)
        self.assertEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
        agent.reduce_state.assert_called_once()
        manager.socket_manager.close_channel.assert_called_once_with(agent.get_connection_id())
        self.assertTrue(agent.disconnected)
        manager._handle_agent_disconnect.assert_not_called()
        agent.disconnected = False
        agent.set_status(AssignState.STATUS_WAITING)
        agent.set_status.reset_mock()
        agent.reduce_state.reset_mock()
        manager.socket_manager.close_channel.reset_mock()
        manager._add_agent_to_pool(agent)
        manager._remove_from_agent_pool = mock.MagicMock()
        manager._on_socket_dead(worker_id, assignment_id)
        self.assertEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
        agent.reduce_state.assert_called_once()
        manager.socket_manager.close_channel.assert_called_once_with(agent.get_connection_id())
        self.assertTrue(agent.disconnected)
        manager._handle_agent_disconnect.assert_not_called()
        manager._remove_from_agent_pool.assert_called_once_with(agent)
        agent.disconnected = False
        agent.set_status(AssignState.STATUS_IN_TASK)
        agent.set_status.reset_mock()
        agent.reduce_state.reset_mock()
        manager.socket_manager.close_channel.reset_mock()
        manager._add_agent_to_pool(agent)
        manager._remove_from_agent_pool = mock.MagicMock()
        manager._on_socket_dead(worker_id, assignment_id)
        self.assertEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
        manager.socket_manager.close_channel.assert_called_once_with(agent.get_connection_id())
        self.assertTrue(agent.disconnected)
        manager._handle_agent_disconnect.assert_called_once_with(worker_id, assignment_id)
        agent.disconnected = False
        agent.set_status(AssignState.STATUS_DONE)
        agent.set_status.reset_mock()
        agent.reduce_state.reset_mock()
        manager._handle_agent_disconnect.reset_mock()
        manager.socket_manager.close_channel.reset_mock()
        manager._add_agent_to_pool(agent)
        manager._remove_from_agent_pool = mock.MagicMock()
        manager._on_socket_dead(worker_id, assignment_id)
        self.assertNotEqual(agent.get_status(), AssignState.STATUS_DISCONNECT)
        agent.reduce_state.assert_not_called()
        manager.socket_manager.close_channel.assert_not_called()
        self.assertFalse(agent.disconnected)
        manager._handle_agent_disconnect.assert_not_called()

    def test_send_message_command(self):
        manager = self.mturk_manager
        worker_id = self.agent_1.worker_id
        assignment_id = self.agent_1.assignment_id
        manager.socket_manager.queue_packet = mock.MagicMock()
        data = {'text': data_model.COMMAND_SEND_MESSAGE}
        manager.send_command(worker_id, assignment_id, data)
        manager.socket_manager.queue_packet.assert_called_once()
        packet = manager.socket_manager.queue_packet.call_args[0][0]
        self.assertIsNotNone(packet.id)
        self.assertEqual(packet.type, data_model.WORLD_MESSAGE)
        self.assertEqual(packet.receiver_id, worker_id)
        self.assertEqual(packet.assignment_id, assignment_id)
        self.assertEqual(packet.data, data)
        self.assertEqual(packet.data['type'], data_model.MESSAGE_TYPE_COMMAND)
        data = {'text': 'This is a test message'}
        manager.socket_manager.queue_packet.reset_mock()
        message_id = manager.send_message(worker_id, assignment_id, data)
        manager.socket_manager.queue_packet.assert_called_once()
        packet = manager.socket_manager.queue_packet.call_args[0][0]
        self.assertIsNotNone(packet.id)
        self.assertEqual(packet.type, data_model.WORLD_MESSAGE)
        self.assertEqual(packet.receiver_id, worker_id)
        self.assertEqual(packet.assignment_id, assignment_id)
        self.assertNotEqual(packet.data, data)
        self.assertEqual(data['text'], packet.data['text'])
        self.assertEqual(packet.data['message_id'], message_id)
        self.assertEqual(packet.data['type'], data_model.MESSAGE_TYPE_ACT)

    def test_free_workers(self):
        manager = self.mturk_manager
        manager.socket_manager.close_channel = mock.MagicMock()
        manager.free_workers([self.agent_1])
        manager.socket_manager.close_channel.assert_called_once_with(self.agent_1.get_connection_id())

    def test_force_expire_hit(self):
        manager = self.mturk_manager
        agent = self.agent_1
        worker_id = agent.worker_id
        assignment_id = agent.assignment_id
        socket_manager = manager.socket_manager
        manager.send_command = mock.MagicMock()
        manager.send_state_change = mock.MagicMock()
        socket_manager.close_channel = mock.MagicMock()
        agent.set_status(AssignState.STATUS_DONE)
        manager.force_expire_hit(worker_id, assignment_id)
        manager.send_command.assert_not_called()
        socket_manager.close_channel.assert_not_called()
        self.assertEqual(agent.get_status(), AssignState.STATUS_DONE)
        agent.set_status(AssignState.STATUS_ONBOARDING)
        manager.send_state_change.reset_mock()
        manager.force_expire_hit(worker_id, assignment_id)
        manager.send_state_change.assert_called_once()
        args = manager.send_state_change.call_args[0]
        used_worker_id, used_assignment_id, data = (args[0], args[1], args[2])
        ack_func = manager.send_state_change.call_args[1]['ack_func']
        ack_func()
        self.assertEqual(worker_id, used_worker_id)
        self.assertEqual(assignment_id, used_assignment_id)
        self.assertEqual(agent.get_status(), AssignState.STATUS_EXPIRED)
        self.assertTrue(agent.hit_is_expired)
        self.assertIsNotNone(data['done_text'])
        socket_manager.close_channel.assert_called_once_with(agent.get_connection_id())
        agent.set_status(AssignState.STATUS_ONBOARDING)
        agent.hit_is_expired = False
        manager.send_state_change.reset_mock()
        socket_manager.close_channel = mock.MagicMock()
        special_disconnect_text = 'You were disconnected as part of a test'
        test_ack_function = mock.MagicMock()
        manager.force_expire_hit(worker_id, assignment_id, text=special_disconnect_text, ack_func=test_ack_function)
        manager.send_state_change.assert_called_once()
        args = manager.send_state_change.call_args[0]
        used_worker_id, used_assignment_id, data = (args[0], args[1], args[2])
        ack_func = manager.send_state_change.call_args[1]['ack_func']
        ack_func()
        self.assertEqual(worker_id, used_worker_id)
        self.assertEqual(assignment_id, used_assignment_id)
        self.assertEqual(agent.get_status(), AssignState.STATUS_EXPIRED)
        self.assertTrue(agent.hit_is_expired)
        self.assertEqual(data['done_text'], special_disconnect_text)
        socket_manager.close_channel.assert_called_once_with(agent.get_connection_id())
        test_ack_function.assert_called()

    def test_get_qualifications(self):
        manager = self.mturk_manager
        mturk_utils = MTurkManagerFile.mturk_utils
        mturk_utils.find_or_create_qualification = mock.MagicMock()
        fake_qual = {'QualificationTypeId': 'fake_qual_id', 'Comparator': 'DoesNotExist', 'ActionsGuarded': 'DiscoverPreviewAndAccept'}
        qualifications = manager.get_qualification_list([fake_qual])
        self.assertListEqual(qualifications, [fake_qual])
        self.assertListEqual(manager.qualifications, [fake_qual])
        mturk_utils.find_or_create_qualification.assert_not_called()
        disconnect_qual_name = 'disconnect_qual_name'
        disconnect_qual_id = 'disconnect_qual_id'
        block_qual_name = 'block_qual_name'
        block_qual_id = 'block_qual_id'
        max_time_qual_name = 'max_time_qual_name'
        max_time_qual_id = 'max_time_qual_id'
        unique_qual_name = 'unique_qual_name'
        unique_qual_id = 'unique_qual_id'

        def return_qualifications(qual_name, _text, _sb):
            if qual_name == disconnect_qual_name:
                return disconnect_qual_id
            if qual_name == block_qual_name:
                return block_qual_id
            if qual_name == max_time_qual_name:
                return max_time_qual_id
            if qual_name == unique_qual_name:
                return unique_qual_id
        mturk_utils.find_or_create_qualification = return_qualifications
        manager.opt['disconnect_qualification'] = disconnect_qual_name
        manager.opt['block_qualification'] = block_qual_name
        manager.opt['max_time_qual'] = max_time_qual_name
        manager.opt['unique_qual_name'] = unique_qual_name
        manager.is_unique = True
        manager.has_time_limit = True
        manager.qualifications = None
        qualifications = manager.get_qualification_list()
        for qual in qualifications:
            self.assertEqual(qual['ActionsGuarded'], 'DiscoverPreviewAndAccept')
            self.assertEqual(qual['Comparator'], 'DoesNotExist')
        for qual_id in [disconnect_qual_id, block_qual_id, max_time_qual_id, unique_qual_id]:
            has_qual = False
            for qual in qualifications:
                if qual['QualificationTypeId'] == qual_id:
                    has_qual = True
                    break
            self.assertTrue(has_qual)
        self.assertListEqual(qualifications, manager.qualifications)

    def test_create_additional_hits(self):
        manager = self.mturk_manager
        manager.opt['hit_title'] = 'test_hit_title'
        manager.opt['hit_description'] = 'test_hit_description'
        manager.opt['hit_keywords'] = 'test_hit_keywords'
        manager.opt['reward'] = 0.1
        mturk_utils = MTurkManagerFile.mturk_utils
        fake_hit = 'fake_hit_type'
        mturk_utils.create_hit_type = mock.MagicMock(return_value=fake_hit)
        mturk_utils.subscribe_to_hits = mock.MagicMock()
        mturk_utils.create_hit_with_hit_type = mock.MagicMock(return_value=('page_url', 'hit_id', 'test_hit_response'))
        manager.server_url = 'test_url'
        manager.task_group_id = 'task_group_id'
        manager.topic_arn = 'topic_arn'
        mturk_chat_url = '{}/chat_index?task_group_id={}'.format(manager.server_url, manager.task_group_id)
        hit_url = manager.create_additional_hits(5)
        mturk_utils.create_hit_type.assert_called_once()
        mturk_utils.subscribe_to_hits.assert_called_with(fake_hit, manager.is_sandbox, manager.topic_arn)
        self.assertEqual(len(mturk_utils.create_hit_with_hit_type.call_args_list), 5)
        mturk_utils.create_hit_with_hit_type.assert_called_with(opt=manager.opt, page_url=mturk_chat_url, hit_type_id=fake_hit, num_assignments=1, is_sandbox=manager.is_sandbox)
        self.assertEqual(len(manager.hit_id_list), 5)
        self.assertEqual(hit_url, 'page_url')

    def test_expire_all_hits(self):
        manager = self.mturk_manager
        incomplete_1 = 'incomplete_1'
        incomplete_2 = 'incomplete_2'
        MTurkManagerFile.mturk_utils.expire_hit = mock.MagicMock()
        manager.hit_id_list = [incomplete_1, incomplete_2]
        manager.expire_all_unassigned_hits()
        expire_calls = MTurkManagerFile.mturk_utils.expire_hit.call_args_list
        self.assertEqual(len(expire_calls), 2)
        for hit in [incomplete_1, incomplete_2]:
            found = False
            for expire_call in expire_calls:
                if expire_call[0][1] == hit:
                    found = True
                    break
            self.assertTrue(found)

    def test_qualification_management(self):
        manager = self.mturk_manager
        test_qual_name = 'test_qual'
        other_qual_name = 'other_qual'
        test_qual_id = 'test_qual_id'
        worker_id = self.agent_1.worker_id
        mturk_utils = MTurkManagerFile.mturk_utils
        success_id = 'Success'

        def find_qualification(qual_name, _sandbox):
            if qual_name == test_qual_name:
                return test_qual_id
            return None
        mturk_utils.find_qualification = find_qualification
        mturk_utils.give_worker_qualification = mock.MagicMock()
        mturk_utils.remove_worker_qualification = mock.MagicMock()
        mturk_utils.find_or_create_qualification = mock.MagicMock(return_value=success_id)
        manager.give_worker_qualification(worker_id, test_qual_name)
        mturk_utils.give_worker_qualification.assert_called_once_with(worker_id, test_qual_id, None, manager.is_sandbox)
        manager.remove_worker_qualification(worker_id, test_qual_name)
        mturk_utils.remove_worker_qualification.assert_called_once_with(worker_id, test_qual_id, manager.is_sandbox, '')
        result = manager.create_qualification(test_qual_name, '')
        self.assertEqual(result, success_id)
        result = manager.create_qualification(test_qual_name, '', False)
        self.assertIsNone(result)
        result = manager.create_qualification(other_qual_name, '')
        self.assertEqual(result, success_id)

    def test_partner_disconnect(self):
        manager = self.mturk_manager
        manager.send_state_change = mock.MagicMock()
        self.agent_1.set_status(AssignState.STATUS_IN_TASK)
        manager._handle_partner_disconnect(self.agent_1)
        self.assertEqual(self.agent_1.get_status(), AssignState.STATUS_PARTNER_DISCONNECT)
        args = manager.send_state_change.call_args[0]
        worker_id, assignment_id = (args[0], args[1])
        self.assertEqual(worker_id, self.agent_1.worker_id)
        self.assertEqual(assignment_id, self.agent_1.assignment_id)

    def test_expire_onboarding(self):
        manager = self.mturk_manager
        manager.force_expire_hit = mock.MagicMock()
        self.agent_2.set_status(AssignState.STATUS_ONBOARDING)
        manager._expire_onboarding_pool()
        manager.force_expire_hit.assert_called_once_with(self.agent_2.worker_id, self.agent_2.assignment_id)