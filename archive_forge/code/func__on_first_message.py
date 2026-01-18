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
def _on_first_message(self, message: Dict[str, Any]):
    """
        Handle first message from player.

        Run when a psid is given that is not paired with any assignment yet.
        Launch an overworld, complete onboarding, etc.

        :param message:
            message sent from agent
        """
    if self.service_reference_id is None:
        self.service_reference_id = message['recipient']['id']
    agent_id = message['sender']['id']
    if self.opt['password'] is not None:
        if message['text'] != self.opt['password']:
            self.observe_message(agent_id, 'Sorry, this conversation bot is password-protected. If you have the password, please send it now.')
            return
    self._launch_overworld(agent_id)