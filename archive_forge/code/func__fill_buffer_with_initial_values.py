import copy
import logging
import math
from typing import Any, Dict, List, Optional
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Space
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def _fill_buffer_with_initial_values(self, data_col: str, view_requirement: ViewRequirement, build_for_inference: bool=False) -> bool:
    """Fills the buffer with the initial values for the given data column.
        for dat_col starting with `state_out`, use the initial states of the policy,
        but for other data columns, create a dummy value based on the view requirement
        space.

        Args:
            data_col: The data column to fill the buffer with.
            view_requirement: The view requirement for the view_col. Normally the view
                requirement for the data column is used and if it does not exist for
                some reason the view requirement for view column is used instead.
            build_for_inference: Whether this is getting called for inference or not.

        returns:
            is_state: True if the data_col is an RNN state, False otherwise.
        """
    try:
        space = self.view_requirements[data_col].space
    except KeyError:
        space = view_requirement.space
    is_state = True
    if data_col.startswith('state_out'):
        if self._enable_new_api_stack:
            self._build_buffers({data_col: self.initial_states})
        else:
            if not self.is_policy_recurrent:
                raise ValueError(f'{data_col} is not available, because the given policy isnot recurrent according to the input model_inital_states.Have you forgotten to return non-empty lists inpolicy.get_initial_states()?')
            state_ind = int(data_col.split('_')[-1])
            self._build_buffers({data_col: self.initial_states[state_ind]})
    else:
        is_state = False
        if build_for_inference:
            if isinstance(space, Space):
                fill_value = get_dummy_batch_for_space(space, batch_size=0)
            else:
                fill_value = space
            self._build_buffers({data_col: fill_value})
    return is_state