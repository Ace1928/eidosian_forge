from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent, create_agent_from_model_file
from parlai.core.worlds import create_task
from parlai.utils.world_logging import WorldLogger
from parlai.utils.misc import TimeLogger
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import math
import json
import random
@register_script('self_chat')
class SelfChat(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return self_chat(self.opt)