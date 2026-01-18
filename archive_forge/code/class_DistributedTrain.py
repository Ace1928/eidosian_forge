import parlai.scripts.train_model as single_train
from parlai.core.script import ParlaiScript
import parlai.utils.distributed as distributed_utils
class DistributedTrain(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        with distributed_utils.slurm_distributed_context(self.opt) as opt:
            self.train_loop = single_train.TrainLoop(opt)
            return self.train_loop.train()