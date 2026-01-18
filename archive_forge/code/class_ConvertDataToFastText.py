from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
import random
import tempfile
import parlai.utils.logging as logging
from parlai.core.script import ParlaiScript, register_script
@register_script('convert_to_fasttext', hidden=True)
class ConvertDataToFastText(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return dump_data(self.opt)