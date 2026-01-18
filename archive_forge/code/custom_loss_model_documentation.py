import numpy as np
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.offline import JsonReader
Calculates a custom loss on top of the given policy_loss(es).

        Args:
            policy_loss (List[TensorType]): The list of already calculated
                policy losses (as many as there are optimizers).
            loss_inputs: Struct of np.ndarrays holding the
                entire train batch.

        Returns:
            List[TensorType]: The altered list of policy losses. In case the
                custom loss should have its own optimizer, make sure the
                returned list is one larger than the incoming policy_loss list.
                In case you simply want to mix in the custom loss into the
                already calculated policy losses, return a list of altered
                policy losses (as done in this example below).
        