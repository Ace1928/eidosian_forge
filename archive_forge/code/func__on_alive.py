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
def _on_alive(self, pkt):
    """
        Update MTurkManager's state when a worker sends an alive packet.

        This asks the socket manager to open a new channel and then handles ensuring the
        worker state is consistent
        """
    shared_utils.print_and_log(logging.DEBUG, 'on_agent_alive: {}'.format(pkt))
    worker_id = pkt.data['worker_id']
    hit_id = pkt.data['hit_id']
    assign_id = pkt.data['assignment_id']
    conversation_id = pkt.data['conversation_id']
    if not assign_id:
        shared_utils.print_and_log(logging.WARN, 'Agent ({}) with no assign_id called alive'.format(worker_id))
        return
    self.socket_manager.open_channel(worker_id, assign_id)
    worker_state = self.worker_manager.worker_alive(worker_id)
    if self.db_logger is not None:
        self.db_logger.log_worker_note(worker_id, assign_id, 'Reconnected with conversation_id {} at {}'.format(conversation_id, time.time()))
    if not worker_state.has_assignment(assign_id):
        completed_assignments = worker_state.completed_assignments()
        max_hits = self.max_hits_per_worker
        if self.is_unique and completed_assignments > 0 or (max_hits != 0 and completed_assignments >= max_hits):
            text = 'You have already participated in this HIT the maximum number of times. This HIT is now expired. Please return the HIT.'
            self.force_expire_hit(worker_id, assign_id, text)
            return
        if not self.accepting_workers:
            self.force_expire_hit(worker_id, assign_id)
            return
        convs = worker_state.active_conversation_count()
        allowed_convs = self.opt['allowed_conversations']
        if allowed_convs > 0 and convs >= allowed_convs:
            text = 'You can participate in only {} of these HITs at once. Please return this HIT and finish your existing HITs before accepting more.'.format(allowed_convs)
            self.force_expire_hit(worker_id, assign_id, text)
            return
        self.worker_manager.assign_task_to_worker(hit_id, assign_id, worker_id)
        if self.db_logger is not None:
            self.db_logger.log_worker_accept_assignment(worker_id, assign_id, hit_id)
        agent = self.worker_manager._get_agent(worker_id, assign_id)
        self._onboard_new_agent(agent)
    else:
        shared_utils.print_and_log(logging.WARN, 'Agent ({}) is reconnecting to {}'.format(worker_id, assign_id))