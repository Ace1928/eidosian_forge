import torch
import random
import os
import signal
import parlai.scripts.train_model as single_train
import parlai.utils.distributed as distributed_utils
from parlai.core.script import ParlaiScript, register_script
@register_script('multiprocessing_train', aliases=['mp_train'], hidden=True)
class MultiProcessTrain(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        port = random.randint(32000, 48000)
        return launch_and_train(self.opt, port)