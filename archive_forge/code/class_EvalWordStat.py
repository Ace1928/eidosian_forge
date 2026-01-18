from parlai.core.params import ParlaiParser
from parlai.core.dict import DictionaryAgent
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.core.metrics import normalize_answer
from parlai.core.logs import TensorboardLogger
from collections import Counter
from parlai.core.script import ParlaiScript, register_script
import copy
import numpy
import random
@register_script('eval_wordstat', hidden=True)
class EvalWordStat(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return eval_wordstat(self.opt)