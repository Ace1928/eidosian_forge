from gymnasium.spaces import Box, Discrete, Space
import numpy as np
from typing import List, Optional, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.tf_utils import get_placeholder
from ray.rllib.utils.typing import FromConfigSpec, ModelConfigDict, TensorType
def _postprocess_tf(self, policy, sample_batch, tf_sess):
    """Calculate states' embeddings and add it to SampleBatch."""
    if self.framework == 'tf':
        obs_embeds = tf_sess.run(self._obs_embeds, feed_dict={self._obs_ph: sample_batch[SampleBatch.OBS]})
    else:
        obs_embeds = tf.stop_gradient(self._encoder_net({SampleBatch.OBS: sample_batch[SampleBatch.OBS]})[0]).numpy()
    sample_batch[SampleBatch.OBS_EMBEDS] = obs_embeds
    return sample_batch