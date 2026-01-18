from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.scripts.train_model import setup_args as train_args
from parlai.scripts.train_model import TrainLoop
import parlai.utils.logging as logging
import cProfile
import io
import pdb
import pstats
@register_script('profile_train', hidden=True)
class ProfileTrain(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return profile(self.opt)