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
def _onboard_function(mturk_agent):
    """
            Onboarding wrapper to set state to onboarding properly.
            """
    if self.get_onboard_world:
        conversation_id = 'o_' + str(uuid.uuid4())
        mturk_agent.set_status(AssignState.STATUS_ONBOARDING, conversation_id=conversation_id, agent_id='onboarding')
        try:
            world = self.get_onboard_world(mturk_agent)
            while not world.episode_done():
                world.parley()
        except AgentTimeoutError:
            self.handle_turker_timeout(mturk_agent.worker_id, mturk_agent.assignment_id)
        except AbsentAgentError:
            pass
        world.shutdown()
        world.review_work()
        save_data = world.prep_save_data([mturk_agent])
        if save_data is not None:
            MTurkDataHandler.save_world_data(save_data, self.task_group_id, conversation_id, sandbox=self.is_sandbox)
        mturk_agent.clear_messages()
    self._move_agents_to_waiting([mturk_agent])