from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
class LecunNormal(init_ops.VarianceScaling):

    def __init__(self, seed=None):
        super(LecunNormal, self).__init__(scale=1.0, mode='fan_in', distribution='truncated_normal', seed=seed)

    def get_config(self):
        return {'seed': self.seed}