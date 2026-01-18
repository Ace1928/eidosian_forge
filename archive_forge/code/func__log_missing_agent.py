import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def _log_missing_agent(self, worker_id, assignment_id):
    """
        Logs when an agent was expected to exist, yet for some reason it didn't.

        If these happen often there is a problem
        """
    shared_utils.print_and_log(logging.WARN, 'Expected to have an agent for {}_{}, yet none was found'.format(worker_id, assignment_id))