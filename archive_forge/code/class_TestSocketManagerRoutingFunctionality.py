import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.agents import AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
class TestSocketManagerRoutingFunctionality(unittest.TestCase):
    ID = 'ID'
    SENDER_ID = 'SENDER_ID'
    ASSIGNMENT_ID = 'ASSIGNMENT_ID'
    DATA = 'DATA'
    CONVERSATION_ID = 'CONVERSATION_ID'
    ACK_FUNCTION = 'ACK_FUNCTION'
    WORLD_ID = '[World_{}]'.format(TASK_GROUP_ID_1)

    def on_alive(self, packet):
        self.alive_packet = packet

    def on_message(self, packet):
        self.message_packet = packet

    def on_worker_death(self, worker_id, assignment_id):
        self.dead_worker_id = worker_id
        self.dead_assignment_id = assignment_id

    def on_server_death(self):
        self.server_died = True

    def setUp(self):
        self.AGENT_ALIVE_PACKET = Packet(MESSAGE_ID_1, data_model.AGENT_ALIVE, self.SENDER_ID, self.WORLD_ID, self.ASSIGNMENT_ID, self.DATA, self.CONVERSATION_ID)
        self.MESSAGE_SEND_PACKET_1 = Packet(MESSAGE_ID_2, data_model.WORLD_MESSAGE, self.WORLD_ID, self.SENDER_ID, self.ASSIGNMENT_ID, self.DATA, self.CONVERSATION_ID)
        self.MESSAGE_SEND_PACKET_2 = Packet(MESSAGE_ID_3, data_model.MESSAGE_BATCH, self.WORLD_ID, self.SENDER_ID, self.ASSIGNMENT_ID, self.DATA, self.CONVERSATION_ID)
        self.MESSAGE_SEND_PACKET_3 = Packet(MESSAGE_ID_4, data_model.MESSAGE_BATCH, self.WORLD_ID, self.SENDER_ID, self.ASSIGNMENT_ID, self.DATA, self.CONVERSATION_ID)
        self.fake_socket = MockSocket()
        time.sleep(0.3)
        self.alive_packet = None
        self.message_packet = None
        self.dead_worker_id = None
        self.dead_assignment_id = None
        self.server_died = False
        self.socket_manager = SocketManager('https://127.0.0.1', self.fake_socket.port, self.on_alive, self.on_message, self.on_worker_death, TASK_GROUP_ID_1, 1, self.on_server_death)

    def tearDown(self):
        self.socket_manager.shutdown()
        self.fake_socket.close()

    def test_init_state(self):
        """
        Ensure all of the initial state of the socket_manager is ready.
        """
        self.assertEqual(self.socket_manager.server_url, 'https://127.0.0.1')
        self.assertEqual(self.socket_manager.port, self.fake_socket.port)
        self.assertEqual(self.socket_manager.alive_callback, self.on_alive)
        self.assertEqual(self.socket_manager.message_callback, self.on_message)
        self.assertEqual(self.socket_manager.socket_dead_callback, self.on_worker_death)
        self.assertEqual(self.socket_manager.task_group_id, TASK_GROUP_ID_1)
        self.assertEqual(self.socket_manager.missed_pongs, 1 + 1 / SocketManager.PING_RATE)
        self.assertIsNotNone(self.socket_manager.ws)
        self.assertTrue(self.socket_manager.keep_running)
        self.assertIsNotNone(self.socket_manager.listen_thread)
        self.assertSetEqual(self.socket_manager.open_channels, set())
        self.assertDictEqual(self.socket_manager.packet_map, {})
        self.assertTrue(self.socket_manager.alive)
        self.assertFalse(self.socket_manager.is_shutdown)
        self.assertEqual(self.socket_manager.get_my_sender_id(), self.WORLD_ID)

    def _send_packet_in_background(self, packet, send_time):
        """
        creates a thread to handle waiting for a packet send.
        """

        def do_send():
            self.socket_manager._send_packet(packet, send_time)
            self.sent = True
        send_thread = threading.Thread(target=do_send, daemon=True)
        send_thread.start()
        time.sleep(0.02)

    def test_packet_send(self):
        """
        Checks to see if packets are working.
        """
        self.socket_manager._safe_send = mock.MagicMock()
        self.sent = False
        send_time = time.time()
        self.assertEqual(self.MESSAGE_SEND_PACKET_2.status, Packet.STATUS_INIT)
        self._send_packet_in_background(self.MESSAGE_SEND_PACKET_2, send_time)
        self.assertEqual(self.MESSAGE_SEND_PACKET_2.status, Packet.STATUS_SENT)
        self.socket_manager._safe_send.assert_called_once()
        self.assertTrue(self.sent)
        used_packet_json = self.socket_manager._safe_send.call_args[0][0]
        used_packet_dict = json.loads(used_packet_json)
        self.assertEqual(used_packet_dict['type'], data_model.MESSAGE_BATCH)
        self.assertDictEqual(used_packet_dict['content'], self.MESSAGE_SEND_PACKET_2.as_dict())

    def test_simple_packet_channel_management(self):
        """
        Ensure that channels are created, managed, and then removed as expected.
        """
        use_packet = self.MESSAGE_SEND_PACKET_1
        worker_id = use_packet.receiver_id
        assignment_id = use_packet.assignment_id
        self.socket_manager.open_channel(worker_id, assignment_id)
        time.sleep(0.1)
        connection_id = use_packet.get_receiver_connection_id()
        self.assertIn(connection_id, self.socket_manager.open_channels)
        self.assertTrue(self.socket_manager.socket_is_open(connection_id))
        self.assertFalse(self.socket_manager.socket_is_open(FAKE_ID))
        resp = self.socket_manager.queue_packet(use_packet)
        self.assertIn(use_packet.id, self.socket_manager.packet_map)
        self.assertTrue(resp)
        self.assertEqual(self.socket_manager.get_status(use_packet.id), use_packet.status)
        self.assertEqual(self.socket_manager.get_status(FAKE_ID), Packet.STATUS_NONE)
        self.socket_manager.close_channel(connection_id)
        time.sleep(0.2)
        self.assertNotIn(connection_id, self.socket_manager.open_channels)
        self.assertNotIn(use_packet.id, self.socket_manager.packet_map)
        self.socket_manager.open_channel(worker_id, assignment_id)
        self.socket_manager.open_channel(worker_id + '2', assignment_id)
        time.sleep(0.1)
        self.assertEqual(len(self.socket_manager.open_channels), 2)
        self.socket_manager.close_all_channels()
        time.sleep(0.1)
        self.assertEqual(len(self.socket_manager.open_channels), 0)