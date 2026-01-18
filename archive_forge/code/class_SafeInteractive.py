from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.safe_local_human.safe_local_human import SafeLocalHumanAgent
import parlai.utils.logging as logging
import random
@register_script('safe_interactive')
class SafeInteractive(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return safe_interactive(self.opt)