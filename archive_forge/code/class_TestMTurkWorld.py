import unittest
import time
import uuid
import os
from unittest import mock
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worlds import MTurkOnboardWorld, MTurkTaskWorld
from parlai.mturk.core.dev.agents import AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.mturk_manager as MTurkManagerFile
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
class TestMTurkWorld(MTurkTaskWorld):

    def __init__(self, workers, use_episode_done):
        self.workers = workers

        def episode_done():
            return use_episode_done()
        self.episode_done = episode_done

    def parley(self):
        for worker in self.workers:
            worker.assert_connected()
        time.sleep(0.5)

    def shutdown(self):
        for worker in self.workers:
            worker.shutdown()