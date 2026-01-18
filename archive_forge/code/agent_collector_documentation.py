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
Fills the buffer with the initial values for the given data column.
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
        