from parlai.core.dict import DictionaryAgent
from parlai.core.params import ParlaiParser, str2class
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.utils.distributed import is_distributed
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import copy
import os
import tqdm
@register_script('build_dict', hidden=True)
class BuildDict(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args(hidden=False)

    def run(self):
        return build_dict(self.opt)