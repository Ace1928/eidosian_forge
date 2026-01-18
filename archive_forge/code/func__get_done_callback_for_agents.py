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
def _get_done_callback_for_agents(self, task_id: str, world_type: str, agents: List[ChatServiceAgent]) -> Callable[[Future], None]:
    """
        Create done callback for finishing task world with particular agents.

        :param task_id:
            task identifier
        :param world_type:
            world name
        :param agents:
            agents for which we are retrieving done callback

        :return:
            the done callback, i.e. the callback function for when agents are done
            in a world.
        """

    def _done_callback(fut):
        """
            Log and raise exception of task world, if there is one.

            Additionally, set active agent to overworld agent.
            """
        e = fut.exception()
        if e is not None:
            log_utils.print_and_log(logging.ERROR, 'World {} had error {}'.format(world_type, repr(e)), should_print=True)
            traceback.print_exc(file=sys.stdout)
            for agent in agents:
                self.observe_message(agent.id, 'Sorry, this world closed. Returning to overworld.')
        else:
            log_utils.print_and_log(logging.INFO, 'World {} had no error'.format(world_type), should_print=True)
        self.active_worlds[task_id] = None
        for agent in agents:
            self.after_agent_removed(agent.id)
            agent_state = self.get_agent_state(agent.id)
            agent_state.data = agent.data
            next_task = agent.data.get('next_task')
            log_utils.print_and_log(logging.INFO, 'Next task: {}'.format(next_task))
            if next_task is None:
                self._launch_overworld(agent.id)
                overworld_agent = agent_state.get_overworld_agent()
                overworld_agent.data = agent_state.data
                agent_state.set_active_agent(overworld_agent)
            elif next_task == self.EXIT_STR:
                self._remove_agent(agent.id)
            else:
                self.add_agent_to_pool(agent_state, next_task)
    return _done_callback