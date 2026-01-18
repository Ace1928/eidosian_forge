import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def _get_agent_from_pkt(self, pkt):
    """
        Get the agent object corresponding to this packet's sender.
        """
    worker_id = pkt.sender_id
    assignment_id = pkt.assignment_id
    agent = self._get_agent(worker_id, assignment_id)
    if agent is None:
        self._log_missing_agent(worker_id, assignment_id)
    return agent