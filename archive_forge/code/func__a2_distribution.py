from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
def _a2_distribution(self, a1):
    a1_vec = torch.unsqueeze(a1.float(), 1)
    _, a2_logits = self.model.action_module(self.inputs, a1_vec)
    a2_dist = TorchCategorical(a2_logits)
    return a2_dist