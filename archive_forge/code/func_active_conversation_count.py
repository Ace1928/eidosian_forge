import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def active_conversation_count(self):
    """
        Return the number of conversations within this worker state that aren't in a
        final state.
        """
    count = 0
    for assign_id in self.agents:
        if not self.agents[assign_id].is_final():
            count += 1
    return count