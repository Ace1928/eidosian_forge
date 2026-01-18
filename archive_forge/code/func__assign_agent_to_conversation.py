import logging
import os
import pickle
import time
from botocore.exceptions import ClientError
from parlai.mturk.core.agents import MTurkAgent
from parlai.mturk.core.shared_utils import AssignState
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.mturk_utils as mturk_utils
import parlai.mturk.core.shared_utils as shared_utils
def _assign_agent_to_conversation(self, agent, conv_id):
    """
        Register an agent object with a conversation id, update status.
        """
    agent.conversation_id = conv_id
    if conv_id not in self.conv_to_agent:
        self.conv_to_agent[conv_id] = []
    self.conv_to_agent[conv_id].append(agent)