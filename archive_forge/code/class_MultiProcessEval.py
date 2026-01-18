import torch
import random
import os
import signal
import parlai.utils.distributed as distributed_utils
import parlai.scripts.eval_model as eval_model
from parlai.core.script import ParlaiScript, register_script
@register_script('multiprocessing_eval', aliases=['mp_eval'], hidden=True)
class MultiProcessEval(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        port = random.randint(32000, 48000)
        return launch_and_eval(self.opt, port)