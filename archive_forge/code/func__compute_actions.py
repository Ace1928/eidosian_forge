import copy
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any
import ray
from ray.data import Dataset
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, convert_ma_batch_to_sample_batch
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
def _compute_actions(batch: pd.DataFrame, policy_state: Dict[str, Any], input_key: str='', output_key: str=''):
    """A custom local function to do batch prediction of a policy.

    Given the policy state the action predictions are computed as a function of
    `input_key` and stored in the `output_key` column.

    Args:
        batch: A sub-batch from the dataset.
        policy_state: The state of the policy to use for the prediction.
        input_key: The key to use for the input to the policy. If not given, the
            default is SampleBatch.OBS.
        output_key: The key to use for the output of the policy. If not given, the
            default is "predicted_actions".

    Returns:
        The modified batch with the predicted actions added as a column.
    """
    if not input_key:
        input_key = SampleBatch.OBS
    policy = Policy.from_state(policy_state)
    sample_batch = SampleBatch({SampleBatch.OBS: np.vstack(batch[input_key].values)})
    actions, _, _ = policy.compute_actions_from_input_dict(sample_batch, explore=False)
    if not output_key:
        output_key = 'predicted_actions'
    batch[output_key] = actions
    return batch