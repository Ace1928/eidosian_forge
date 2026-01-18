from abc import ABC, abstractmethod
from asyncio import Future
import copy
import sys
import logging
import datetime
import threading
import time
import traceback
from typing import Dict, Any, Optional, List, Callable
from parlai.chat_service.core.agents import ChatServiceAgent
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
import parlai.chat_service.utils.server as server_utils
from parlai.chat_service.core.world_runner import ChatServiceWorldRunner
from parlai.core.opt import Opt
def add_agent_to_pool(self, agent: AgentState, world_type: str='default'):
    """
        Add the agent to pool.

        :param agent:
            AgentState object
        :param world_type:
            Name of world whose pool should now contain agent
        """
    with self.agent_pool_change_condition:
        self._log_debug('Adding agent {} to pool...'.format(agent.service_id))
        agent.time_in_pool.setdefault(world_type, time.time())
        self.agent_pool.setdefault(world_type, []).append(agent)