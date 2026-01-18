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
def _on_socket_dead(self, worker_id, assignment_id):
    """
        Handle a disconnect event, update state as required and notifying other agents
        if the disconnected agent was in conversation with them.

        returns False if the socket death should be ignored and the socket should stay
        open and not be considered disconnected
        """
    agent = self.worker_manager._get_agent(worker_id, assignment_id)
    if agent is None:
        return
    shared_utils.print_and_log(logging.DEBUG, 'Worker {} disconnected from {} in status {}'.format(worker_id, agent.conversation_id, agent.get_status()))
    if agent.get_status() == AssignState.STATUS_NONE:
        agent.set_status(AssignState.STATUS_DISCONNECT)
        agent.reduce_state()
    elif agent.get_status() == AssignState.STATUS_ONBOARDING:
        agent.set_status(AssignState.STATUS_DISCONNECT)
        agent.reduce_state()
        agent.disconnected = True
    elif agent.get_status() == AssignState.STATUS_WAITING:
        if agent in self.agent_pool:
            with self.agent_pool_change_condition:
                self._remove_from_agent_pool(agent)
        agent.set_status(AssignState.STATUS_DISCONNECT)
        agent.reduce_state()
        agent.disconnected = True
    elif agent.get_status() == AssignState.STATUS_IN_TASK:
        self._handle_agent_disconnect(worker_id, assignment_id)
        agent.disconnected = True
    elif agent.get_status() == AssignState.STATUS_DONE:
        return
    self.socket_manager.close_channel(agent.get_connection_id())