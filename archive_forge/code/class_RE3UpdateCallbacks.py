import gc
import os
import platform
import tracemalloc
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.exploration.random_encoder import (
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from ray.tune.callback import _CallbackMeta
import psutil
class RE3UpdateCallbacks(DefaultCallbacks):
    """Update input callbacks to mutate batch with states entropy rewards."""
    _step = 0

    def __init__(self, *args, embeds_dim: int=128, k_nn: int=50, beta: float=0.1, rho: float=0.0001, beta_schedule: str='constant', **kwargs):
        self.embeds_dim = embeds_dim
        self.k_nn = k_nn
        self.beta = beta
        self.rho = rho
        self.beta_schedule = beta_schedule
        self._rms = _MovingMeanStd()
        super().__init__(*args, **kwargs)

    @override(DefaultCallbacks)
    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs):
        super().on_learn_on_batch(policy=policy, train_batch=train_batch, result=result, **kwargs)
        states_entropy = compute_states_entropy(train_batch[SampleBatch.OBS_EMBEDS], self.embeds_dim, self.k_nn)
        states_entropy = update_beta(self.beta_schedule, self.beta, self.rho, RE3UpdateCallbacks._step) * np.reshape(self._rms(states_entropy), train_batch[SampleBatch.OBS_EMBEDS].shape[:-1])
        train_batch[SampleBatch.REWARDS] = train_batch[SampleBatch.REWARDS] + states_entropy
        if Postprocessing.ADVANTAGES in train_batch:
            train_batch[Postprocessing.ADVANTAGES] = train_batch[Postprocessing.ADVANTAGES] + states_entropy
            train_batch[Postprocessing.VALUE_TARGETS] = train_batch[Postprocessing.VALUE_TARGETS] + states_entropy

    @override(DefaultCallbacks)
    def on_train_result(self, *, result: dict, algorithm=None, **kwargs) -> None:
        RE3UpdateCallbacks._step = result['training_iteration']
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)