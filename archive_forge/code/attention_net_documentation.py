import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, Optional, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules import (
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor, one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType, List
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util import log_once
Initializes a GTrXLNet.

        Args:
            num_transformer_units: The number of Transformer repeats to
                use (denoted L in [2]).
            attention_dim: The input and output dimensions of one
                Transformer unit.
            num_heads: The number of attention heads to use in parallel.
                Denoted as `H` in [3].
            memory_inference: The number of timesteps to concat (time
                axis) and feed into the next transformer unit as inference
                input. The first transformer unit will receive this number of
                past observations (plus the current one), instead.
            memory_training: The number of timesteps to concat (time
                axis) and feed into the next transformer unit as training
                input (plus the actual input sequence of len=max_seq_len).
                The first transformer unit will receive this number of
                past observations (plus the input sequence), instead.
            head_dim: The dimension of a single(!) attention head within
                a multi-head attention unit. Denoted as `d` in [3].
            position_wise_mlp_dim: The dimension of the hidden layer
                within the position-wise MLP (after the multi-head attention
                block within one Transformer unit). This is the size of the
                first of the two layers within the PositionwiseFeedforward. The
                second layer always has size=`attention_dim`.
            init_gru_gate_bias: Initial bias values for the GRU gates
                (two GRUs per Transformer unit, one after the MHA, one after
                the position-wise MLP).
        