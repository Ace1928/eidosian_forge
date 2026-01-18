from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.utils.misc import TimeLogger, warn_once
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
@register_script('verify_data', hidden=True)
class VerifyData(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return verify_data(self.opt, self.parser)