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
class AgentState:
    """
    Keep track of Agent State.

    State includes which is the "active" agent - i.e., which agent in which
    world do we message, etc.
    """

    def __init__(self, service_id: int, overworld_agent: ChatServiceAgent):
        self.service_id = service_id
        self.overworld_agent = overworld_agent
        self.active_agent = overworld_agent
        self.task_id_to_agent: Dict[str, ChatServiceAgent] = {}
        self.onboard_data = None
        self.data = {}
        self.stored_data: Dict[str, Any] = {}
        self.time_in_pool: Dict[str, float] = {}

    def get_active_agent(self) -> ChatServiceAgent:
        """
        Return active messenger agent.

        :return:
            a ChatServiceAgent, which corresponds to the active agent for this
            agent state.
        """
        return self.active_agent

    def set_active_agent(self, active_agent: ChatServiceAgent):
        """
        Set active agent for this agent.

        :param active_agent:
            A ChatServiceAgent, the new active agent for this given agent state
        """
        self.active_agent = active_agent

    def get_overworld_agent(self) -> ChatServiceAgent:
        """
        Return overworld messenger agent.

        :return:
            a ChatServiceAgent, which corresponds agent object in the overworld
        """
        return self.overworld_agent

    def get_id(self) -> int:
        """
        Return the agent's ID.

        :return:
            int agent's service ID
        """
        return self.service_id

    def has_task(self, task_id: str) -> bool:
        """
        Determine if an agent is in a task.

        :param task_id:
            task id

        :return:
            if agent is in that task.
        """
        return task_id in self.task_id_to_agent

    def get_agent_for_task(self, task_id: str) -> Optional[ChatServiceAgent]:
        """
        Return ChatServiceAgent for given task id.

        For each "player", a separate agent is created for each task. This
        returns the appropriate MessengerAgent given the task id

        :param task_id:
            task id

        :return:
            ChatServiceAgent object associated with the given task
        """
        if self.has_task(task_id):
            return self.task_id_to_agent[task_id]
        else:
            return None

    def assign_agent_to_task(self, agent: ChatServiceAgent, task_id: str):
        """
        Mark agent in task.

        :param agent:
            ChatServiceAgent object to mark in task
        :param task_id:
            string task name
        """
        self.task_id_to_agent[task_id] = agent