from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.repeat_query.repeat_query import RepeatQueryAgent
import parlai.utils.logging as logging
import random
import cProfile
import io
import pstats
@register_script('profile_interactive', hidden=True)
class ProfileInteractive(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return profile_interactive(self.opt)