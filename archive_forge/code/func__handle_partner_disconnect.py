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
def _handle_partner_disconnect(self, agent):
    """
        Send a message to an agent notifying them that a partner has disconnected and we
        marked the HIT as complete for them.
        """
    if agent is not None and (not agent.is_final()):
        agent.some_agent_disconnected = True
        agent_messages = [m for m in agent.get_messages() if 'id' in m and m['id'] == agent.id]
        if len(agent_messages) < self.minimum_messages:
            agent.set_status(AssignState.STATUS_PARTNER_DISCONNECT_EARLY)
        else:
            agent.set_status(AssignState.STATUS_PARTNER_DISCONNECT)
        data = {'agent_status': AssignState.STATUS_PARTNER_DISCONNECT, 'done_text': "One of your partners disconnected in the middle of the HIT. We won't penalize you for their disconnect, so please use the button below to mark the HIT as complete."}

        def disconnect_agent(*args):
            self.socket_manager.close_channel(agent.get_connection_id())
        self.send_state_change(agent.worker_id, agent.assignment_id, data, ack_func=disconnect_agent)