import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
class AgentDisconnectedError(AbsentAgentError):
    """
    Exception for a real disconnect event (no signal)
    """

    def __init__(self, worker_id, assignment_id):
        super().__init__(f'Agent disconnected', worker_id, assignment_id)