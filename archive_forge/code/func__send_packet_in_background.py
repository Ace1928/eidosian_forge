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