from collections import namedtuple, OrderedDict
import gymnasium as gym
import logging
import re
import tree  # pip install dm_tree
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from ray.util.debug import log_once
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.spaces.space_utils import get_dummy_batch_for_space
from ray.rllib.utils.tf_utils import get_placeholder
from ray.rllib.utils.typing import (
def _get_input_dict_and_dummy_batch(self, view_requirements, existing_inputs):
    """Creates input_dict and dummy_batch for loss initialization.

        Used for managing the Policy's input placeholders and for loss
        initialization.
        Input_dict: Str -> tf.placeholders, dummy_batch: str -> np.arrays.

        Args:
            view_requirements: The view requirements dict.
            existing_inputs (Dict[str, tf.placeholder]): A dict of already
                existing placeholders.

        Returns:
            Tuple[Dict[str, tf.placeholder], Dict[str, np.ndarray]]: The
                input_dict/dummy_batch tuple.
        """
    input_dict = {}
    for view_col, view_req in view_requirements.items():
        mo = re.match('state_in_(\\d+)', view_col)
        if mo is not None:
            input_dict[view_col] = self._state_inputs[int(mo.group(1))]
        elif view_col.startswith('state_out_'):
            continue
        elif view_col == SampleBatch.ACTION_DIST_INPUTS:
            continue
        elif view_col in existing_inputs:
            input_dict[view_col] = existing_inputs[view_col]
        else:
            time_axis = not isinstance(view_req.shift, int)
            if view_req.used_for_training:
                if self.config.get('_disable_action_flattening') and view_col in [SampleBatch.ACTIONS, SampleBatch.PREV_ACTIONS]:
                    flatten = False
                elif view_col in [SampleBatch.OBS, SampleBatch.NEXT_OBS] and self.config['_disable_preprocessor_api']:
                    flatten = False
                else:
                    flatten = True
                input_dict[view_col] = get_placeholder(space=view_req.space, name=view_col, time_axis=time_axis, flatten=flatten)
    dummy_batch = self._get_dummy_batch_from_view_requirements(batch_size=32)
    return (SampleBatch(input_dict, seq_lens=self._seq_lens), dummy_batch)