import glob
import json
import logging
import math
import numpy as np
import os
from pathlib import Path
import random
import re
import tree  # pip install dm_tree
from typing import List, Optional, TYPE_CHECKING, Union
from urllib.parse import urlparse
import zipfile
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import (
from ray.rllib.utils.annotations import override, PublicAPI, DeveloperAPI
from ray.rllib.utils.compression import unpack_if_needed
from ray.rllib.utils.spaces.space_utils import clip_action, normalize_action
from ray.rllib.utils.typing import Any, FileType, SampleBatchType
def _adjust_obs_actions_for_policy(json_data: dict, policy: Policy) -> dict:
    """Handle nested action/observation spaces for policies.

    Translates nested lists/dicts from the json into proper
    np.ndarrays, according to the (nested) observation- and action-
    spaces of the given policy.

    Providing nested lists w/o this preprocessing step would
    confuse a SampleBatch constructor.
    """
    for k, v in json_data.items():
        data_col = policy.view_requirements[k].data_col if k in policy.view_requirements else ''
        if policy.config.get('_disable_action_flattening') and (k == SampleBatch.ACTIONS or data_col == SampleBatch.ACTIONS or k == SampleBatch.PREV_ACTIONS or (data_col == SampleBatch.PREV_ACTIONS)):
            json_data[k] = tree.map_structure_up_to(policy.action_space_struct, lambda comp: np.array(comp), json_data[k], check_types=False)
        elif policy.config.get('_disable_preprocessor_api') and (k == SampleBatch.OBS or data_col == SampleBatch.OBS or k == SampleBatch.NEXT_OBS or (data_col == SampleBatch.NEXT_OBS)):
            json_data[k] = tree.map_structure_up_to(policy.observation_space_struct, lambda comp: np.array(comp), json_data[k], check_types=False)
    return json_data