import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
class AbsentAgentError(Exception):
    """
    Exceptions for when an agent leaves a task.
    """

    def __init__(self, message, worker_id, assignment_id):
        self.message = message
        self.worker_id = worker_id
        self.assignment_id = assignment_id