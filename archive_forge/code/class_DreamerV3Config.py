import copy
import dataclasses
import gc
import logging
import tree  # pip install dm_tree
from typing import Any, Dict, List, Optional, Union
import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dreamerv3.dreamerv3_catalog import DreamerV3Catalog
from ray.rllib.algorithms.dreamerv3.dreamerv3_learner import (
from ray.rllib.algorithms.dreamerv3.utils import do_symlog_obs
from ray.rllib.algorithms.dreamerv3.utils.env_runner import DreamerV3EnvRunner
from ray.rllib.algorithms.dreamerv3.utils.summaries import (
from ray.rllib.core.learner.learner import LearnerHyperparameters
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.models.catalog import MODEL_DEFAULTS
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.numpy import one_hot
from ray.rllib.utils.metrics import (
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer
from ray.rllib.utils.typing import LearningRateOrSchedule, ResultDict
class DreamerV3Config(AlgorithmConfig):
    """Defines a configuration class from which a DreamerV3 can be built.

    .. testcode::

        from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
        config = (
            DreamerV3Config()
            .environment("CartPole-v1")
            .training(
                model_size="XS",
                training_ratio=1,
                # TODO
                model={
                    "batch_size_B": 1,
                    "batch_length_T": 1,
                    "horizon_H": 1,
                    "gamma": 0.997,
                    "model_size": "XS",
                },
            )
        )

        config = config.resources(num_learner_workers=0)
        # Build a Algorithm object from the config and run 1 training iteration.
        algo = config.build()
        # algo.train()
        del algo

    .. testoutput::
        :hide:

        ...
    """

    def __init__(self, algo_class=None):
        """Initializes a DreamerV3Config instance."""
        super().__init__(algo_class=algo_class or DreamerV3)
        self.model_size = 'XS'
        self.training_ratio = 1024
        self.replay_buffer_config = {'type': 'EpisodeReplayBuffer', 'capacity': int(1000000.0)}
        self.world_model_lr = 0.0001
        self.actor_lr = 3e-05
        self.critic_lr = 3e-05
        self.batch_size_B = 16
        self.batch_length_T = 64
        self.horizon_H = 15
        self.gae_lambda = 0.95
        self.entropy_scale = 0.0003
        self.return_normalization_decay = 0.99
        self.train_critic = True
        self.train_actor = True
        self.intrinsic_rewards_scale = 0.1
        self.world_model_grad_clip_by_global_norm = 1000.0
        self.critic_grad_clip_by_global_norm = 100.0
        self.actor_grad_clip_by_global_norm = 100.0
        self.symlog_obs = 'auto'
        self.use_float16 = False
        self.metrics_num_episodes_for_smoothing = 1
        self.report_individual_batch_item_stats = False
        self.report_dream_data = False
        self.report_images_and_videos = False
        self.gc_frequency_train_steps = 100
        self.lr = None
        self.framework_str = 'tf2'
        self.gamma = 0.997
        self.train_batch_size = None
        self.env_runner_cls = DreamerV3EnvRunner
        self.num_rollout_workers = 0
        self.rollout_fragment_length = 1
        self.remote_worker_envs = True
        self._enable_new_api_stack = True

    @property
    def model(self):
        model = copy.deepcopy(MODEL_DEFAULTS)
        model.update({'batch_length_T': self.batch_length_T, 'gamma': self.gamma, 'horizon_H': self.horizon_H, 'model_size': self.model_size, 'symlog_obs': self.symlog_obs, 'use_float16': self.use_float16})
        return model

    @override(AlgorithmConfig)
    def training(self, *, model_size: Optional[str]=NotProvided, training_ratio: Optional[float]=NotProvided, gc_frequency_train_steps: Optional[int]=NotProvided, batch_size_B: Optional[int]=NotProvided, batch_length_T: Optional[int]=NotProvided, horizon_H: Optional[int]=NotProvided, gae_lambda: Optional[float]=NotProvided, entropy_scale: Optional[float]=NotProvided, return_normalization_decay: Optional[float]=NotProvided, train_critic: Optional[bool]=NotProvided, train_actor: Optional[bool]=NotProvided, intrinsic_rewards_scale: Optional[float]=NotProvided, world_model_lr: Optional[LearningRateOrSchedule]=NotProvided, actor_lr: Optional[LearningRateOrSchedule]=NotProvided, critic_lr: Optional[LearningRateOrSchedule]=NotProvided, world_model_grad_clip_by_global_norm: Optional[float]=NotProvided, critic_grad_clip_by_global_norm: Optional[float]=NotProvided, actor_grad_clip_by_global_norm: Optional[float]=NotProvided, symlog_obs: Optional[Union[bool, str]]=NotProvided, use_float16: Optional[bool]=NotProvided, replay_buffer_config: Optional[dict]=NotProvided, **kwargs) -> 'DreamerV3Config':
        """Sets the training related configuration.

        Args:
            model_size: The main switch for adjusting the overall model size. See [1]
                (table B) for more information on the effects of this setting on the
                model architecture.
                Supported values are "XS", "S", "M", "L", "XL" (as per the paper), as
                well as, "nano", "micro", "mini", and "XXS" (for RLlib's
                implementation). See ray.rllib.algorithms.dreamerv3.utils.
                __init__.py for the details on what exactly each size does to the layer
                sizes, number of layers, etc..
            training_ratio: The ratio of total steps trained (sum of the sizes of all
                batches ever sampled from the replay buffer) over the total env steps
                taken (in the actual environment, not the dreamed one). For example,
                if the training_ratio is 1024 and the batch size is 1024, we would take
                1 env step for every training update: 1024 / 1. If the training ratio
                is 512 and the batch size is 1024, we would take 2 env steps and then
                perform a single training update (on a 1024 batch): 1024 / 2.
            gc_frequency_train_steps: The frequency (in training iterations) with which
                we perform a `gc.collect()` calls at the end of a `training_step`
                iteration. Doing this more often adds a (albeit very small) performance
                overhead, but prevents memory leaks from becoming harmful.
                TODO (sven): This might not be necessary anymore, but needs to be
                 confirmed experimentally.
            batch_size_B: The batch size (B) interpreted as number of rows (each of
                length `batch_length_T`) to sample from the replay buffer in each
                iteration.
            batch_length_T: The batch length (T) interpreted as the length of each row
                sampled from the replay buffer in each iteration. Note that
                `batch_size_B` rows will be sampled in each iteration. Rows normally
                contain consecutive data (consecutive timesteps from the same episode),
                but there might be episode boundaries in a row as well.
            horizon_H: The horizon (in timesteps) used to create dreamed data from the
                world model, which in turn is used to train/update both actor- and
                critic networks.
            gae_lambda: The lambda parameter used for computing the GAE-style
                value targets for the actor- and critic losses.
            entropy_scale: The factor with which to multiply the entropy loss term
                inside the actor loss.
            return_normalization_decay: The decay value to use when computing the
                running EMA values for return normalization (used in the actor loss).
            train_critic: Whether to train the critic network. If False, `train_actor`
                must also be False (cannot train actor w/o training the critic).
            train_actor: Whether to train the actor network. If True, `train_critic`
                must also be True (cannot train actor w/o training the critic).
            intrinsic_rewards_scale: The factor to multiply intrinsic rewards with
                before adding them to the extrinsic (environment) rewards.
            world_model_lr: The learning rate or schedule for the world model optimizer.
            actor_lr: The learning rate or schedule for the actor optimizer.
            critic_lr: The learning rate or schedule for the critic optimizer.
            world_model_grad_clip_by_global_norm: World model grad clipping value
                (by global norm).
            critic_grad_clip_by_global_norm: Critic grad clipping value
                (by global norm).
            actor_grad_clip_by_global_norm: Actor grad clipping value (by global norm).
            symlog_obs: Whether to symlog observations or not. If set to "auto"
                (default), will check for the environment's observation space and then
                only symlog if not an image space.
            use_float16: Whether to train with mixed float16 precision. In this mode,
                model parameters are stored as float32, but all computations are
                performed in float16 space (except for losses and distribution params
                and outputs).
            replay_buffer_config: Replay buffer config.
                Only serves in DreamerV3 to set the capacity of the replay buffer.
                Note though that in the paper ([1]) a size of 1M is used for all
                benchmarks and there doesn't seem to be a good reason to change this
                parameter.
                Examples:
                {
                "type": "EpisodeReplayBuffer",
                "capacity": 100000,
                }

        Returns:
            This updated AlgorithmConfig object.
        """
        super().training(**kwargs)
        if model_size is not NotProvided:
            self.model_size = model_size
        if training_ratio is not NotProvided:
            self.training_ratio = training_ratio
        if gc_frequency_train_steps is not NotProvided:
            self.gc_frequency_train_steps = gc_frequency_train_steps
        if batch_size_B is not NotProvided:
            self.batch_size_B = batch_size_B
        if batch_length_T is not NotProvided:
            self.batch_length_T = batch_length_T
        if horizon_H is not NotProvided:
            self.horizon_H = horizon_H
        if gae_lambda is not NotProvided:
            self.gae_lambda = gae_lambda
        if entropy_scale is not NotProvided:
            self.entropy_scale = entropy_scale
        if return_normalization_decay is not NotProvided:
            self.return_normalization_decay = return_normalization_decay
        if train_critic is not NotProvided:
            self.train_critic = train_critic
        if train_actor is not NotProvided:
            self.train_actor = train_actor
        if intrinsic_rewards_scale is not NotProvided:
            self.intrinsic_rewards_scale = intrinsic_rewards_scale
        if world_model_lr is not NotProvided:
            self.world_model_lr = world_model_lr
        if actor_lr is not NotProvided:
            self.actor_lr = actor_lr
        if critic_lr is not NotProvided:
            self.critic_lr = critic_lr
        if world_model_grad_clip_by_global_norm is not NotProvided:
            self.world_model_grad_clip_by_global_norm = world_model_grad_clip_by_global_norm
        if critic_grad_clip_by_global_norm is not NotProvided:
            self.critic_grad_clip_by_global_norm = critic_grad_clip_by_global_norm
        if actor_grad_clip_by_global_norm is not NotProvided:
            self.actor_grad_clip_by_global_norm = actor_grad_clip_by_global_norm
        if symlog_obs is not NotProvided:
            self.symlog_obs = symlog_obs
        if use_float16 is not NotProvided:
            self.use_float16 = use_float16
        if replay_buffer_config is not NotProvided:
            new_replay_buffer_config = deep_update({'replay_buffer_config': self.replay_buffer_config}, {'replay_buffer_config': replay_buffer_config}, False, ['replay_buffer_config'], ['replay_buffer_config'])
            self.replay_buffer_config = new_replay_buffer_config['replay_buffer_config']
        return self

    @override(AlgorithmConfig)
    def reporting(self, *, report_individual_batch_item_stats: Optional[bool]=NotProvided, report_dream_data: Optional[bool]=NotProvided, report_images_and_videos: Optional[bool]=NotProvided, **kwargs):
        """Sets the reporting related configuration.

        Args:
            report_individual_batch_item_stats: Whether to include loss and other stats
                per individual timestep inside the training batch in the result dict
                returned by `training_step()`. If True, besides the `CRITIC_L_total`,
                the individual critic loss values per batch row and time axis step
                in the train batch (CRITIC_L_total_B_T) will also be part of the
                results.
            report_dream_data:  Whether to include the dreamed trajectory data in the
                result dict returned by `training_step()`. If True, however, will
                slice each reported item in the dream data down to the shape.
                (H, B, t=0, ...), where H is the horizon and B is the batch size. The
                original time axis will only be represented by the first timestep
                to not make this data too large to handle.
            report_images_and_videos: Whether to include any image/video data in the
                result dict returned by `training_step()`.
            **kwargs:

        Returns:
            This updated AlgorithmConfig object.
        """
        super().reporting(**kwargs)
        if report_individual_batch_item_stats is not NotProvided:
            self.report_individual_batch_item_stats = report_individual_batch_item_stats
        if report_dream_data is not NotProvided:
            self.report_dream_data = report_dream_data
        if report_images_and_videos is not NotProvided:
            self.report_images_and_videos = report_images_and_videos
        return self

    @override(AlgorithmConfig)
    def validate(self) -> None:
        super().validate()
        if self.is_multi_agent():
            raise ValueError('DreamerV3 does NOT support multi-agent setups yet!')
        if not self._enable_new_api_stack:
            raise ValueError('DreamerV3 must be run with `config.experimental(_enable_new_api_stack=True)`!')
        if self.num_learner_workers > 1 and self.batch_size_B % self.num_learner_workers != 0:
            raise ValueError(f'Your `batch_size_B` ({self.batch_size_B}) must be a multiple of `num_learner_workers` ({self.num_learner_workers}) in order for DreamerV3 to be able to split batches evenly across your Learner processes.')
        if self.train_actor and (not self.train_critic):
            raise ValueError('Cannot train actor network (`train_actor=True`) w/o training critic! Make sure you either set `train_critic=True` or `train_actor=False`.')
        if self.train_batch_size is not None:
            raise ValueError('`train_batch_size` should NOT be set! Use `batch_size_B` and `batch_length_T` instead.')
        if self.replay_buffer_config.get('type') != 'EpisodeReplayBuffer':
            raise ValueError('DreamerV3 must be run with the `EpisodeReplayBuffer` type! None other supported.')

    @override(AlgorithmConfig)
    def get_learner_hyperparameters(self) -> LearnerHyperparameters:
        base_hps = super().get_learner_hyperparameters()
        return DreamerV3LearnerHyperparameters(model_size=self.model_size, training_ratio=self.training_ratio, batch_size_B=self.batch_size_B // (self.num_learner_workers or 1), batch_length_T=self.batch_length_T, horizon_H=self.horizon_H, gamma=self.gamma, gae_lambda=self.gae_lambda, entropy_scale=self.entropy_scale, return_normalization_decay=self.return_normalization_decay, train_actor=self.train_actor, train_critic=self.train_critic, world_model_lr=self.world_model_lr, intrinsic_rewards_scale=self.intrinsic_rewards_scale, actor_lr=self.actor_lr, critic_lr=self.critic_lr, world_model_grad_clip_by_global_norm=self.world_model_grad_clip_by_global_norm, actor_grad_clip_by_global_norm=self.actor_grad_clip_by_global_norm, critic_grad_clip_by_global_norm=self.critic_grad_clip_by_global_norm, use_float16=self.use_float16, report_individual_batch_item_stats=self.report_individual_batch_item_stats, report_dream_data=self.report_dream_data, report_images_and_videos=self.report_images_and_videos, **dataclasses.asdict(base_hps))

    @override(AlgorithmConfig)
    def get_default_learner_class(self):
        if self.framework_str == 'tf2':
            from ray.rllib.algorithms.dreamerv3.tf.dreamerv3_tf_learner import DreamerV3TfLearner
            return DreamerV3TfLearner
        else:
            raise ValueError(f'The framework {self.framework_str} is not supported.')

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> SingleAgentRLModuleSpec:
        if self.framework_str == 'tf2':
            from ray.rllib.algorithms.dreamerv3.tf.dreamerv3_tf_rl_module import DreamerV3TfRLModule
            return SingleAgentRLModuleSpec(module_class=DreamerV3TfRLModule, catalog_class=DreamerV3Catalog)
        else:
            raise ValueError(f'The framework {self.framework_str} is not supported.')

    @property
    def share_module_between_env_runner_and_learner(self) -> bool:
        return self.num_learner_workers == 0 and self.num_rollout_workers == 0