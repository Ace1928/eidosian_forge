from typing import Any, Mapping
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.rl_module.marl_module import (
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict
An example of an RLModule that uses an encoder shared with other things.

    For example, we could consider a multi-agent case where for inference each agent
    needs to know the global state of the environment, as well as the local state of
    itself. For better representation learning we would like to share the encoder
    across all the modules. So this module simply accepts the encoder object as its
    input argument and uses it to encode the global state. The local state is passed
    through as is. The policy head is then a simple MLP that takes the concatenation of
    the global and local state as input and outputs the action logits.

    