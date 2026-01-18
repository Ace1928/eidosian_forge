from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.core.dict import DictionaryAgent
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
@register_script('data_stats', hidden=True)
class DataStats(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return obtain_stats(self.opt, self.parser)