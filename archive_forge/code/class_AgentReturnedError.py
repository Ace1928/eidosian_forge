import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
class AgentReturnedError(AbsentAgentError):
    """
    Exception for an explicit return event (worker returns task)
    """

    def __init__(self, worker_id, assignment_id):
        super().__init__(f'Agent returned HIT', worker_id, assignment_id)