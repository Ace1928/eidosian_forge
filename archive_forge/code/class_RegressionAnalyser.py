import numpy as np
from autokeras.engine import analyser
class RegressionAnalyser(TargetAnalyser):

    def __init__(self, output_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def finalize(self):
        if self.output_dim and self.expected_dim() != self.output_dim:
            raise ValueError('Expect the target data for {name} to have shape (batch_size, {output_dim}), but got {shape}.'.format(name=self.name, output_dim=self.output_dim, shape=self.shape))

    def expected_dim(self):
        if len(self.shape) == 1:
            return 1
        return self.shape[1]