import copy
import dataclasses
from functools import partial
import logging
import platform
import queue
import random
from typing import Callable, List, Optional, Set, Tuple, Type, Union
import numpy as np
import tree  # pip install dm_tree
import ray
from ray import ObjectRef
from ray.rllib import SampleBatch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.impala.impala_learner import (
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.evaluation.worker_set import handle_remote_call_result_errors
from ray.rllib.execution.buffers.mixin_replay_buffer import MixInMultiAgentReplayBuffer
from ray.rllib.execution.learner_thread import LearnerThread
from ray.rllib.execution.multi_gpu_learner_thread import MultiGPULearnerThread
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import concat_samples
from ray.rllib.utils.actor_manager import (
from ray.rllib.utils.actors import create_colocated_actors
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import ReplayMode
from ray.rllib.utils.replay_buffers.replay_buffer import _ALL_POLICIES
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import (
from ray.tune.execution.placement_groups import PlacementGroupFactory
class ImpalaConfig(AlgorithmConfig):
    """Defines a configuration class from which an Impala can be built.

    .. testcode::

        from ray.rllib.algorithms.impala import ImpalaConfig
        config = ImpalaConfig()
        config = config.training(lr=0.0003, train_batch_size=512)
        config = config.resources(num_gpus=0)
        config = config.rollouts(num_rollout_workers=1)
        # Build a Algorithm object from the config and run 1 training iteration.
        algo = config.build(env="CartPole-v1")
        algo.train()
        del algo

    .. testcode::

        from ray.rllib.algorithms.impala import ImpalaConfig
        from ray import air
        from ray import tune
        config = ImpalaConfig()

        # Update the config object.
        config = config.training(
            lr=tune.grid_search([0.0001, ]), grad_clip=20.0
        )
        config = config.resources(num_gpus=0)
        config = config.rollouts(num_rollout_workers=1)
        # Set the config object's env.
        config = config.environment(env="CartPole-v1")
        # Use to_dict() to get the old-style python config dict
        # when running with tune.
        tune.Tuner(
            "IMPALA",
            run_config=air.RunConfig(stop={"training_iteration": 1}),
            param_space=config.to_dict(),
        ).fit()

    .. testoutput::
        :hide:

        ...
    """

    def __init__(self, algo_class=None):
        """Initializes a ImpalaConfig instance."""
        super().__init__(algo_class=algo_class or Impala)
        self.vtrace = True
        self.vtrace_clip_rho_threshold = 1.0
        self.vtrace_clip_pg_rho_threshold = 1.0
        self.num_multi_gpu_tower_stacks = 1
        self.minibatch_buffer_size = 1
        self.num_sgd_iter = 1
        self.replay_proportion = 0.0
        self.replay_buffer_num_slots = 0
        self.learner_queue_size = 16
        self.learner_queue_timeout = 300
        self.max_requests_in_flight_per_aggregator_worker = 2
        self.timeout_s_sampler_manager = 0.0
        self.timeout_s_aggregator_manager = 0.0
        self.broadcast_interval = 1
        self.num_aggregation_workers = 0
        self.grad_clip = 40.0
        self.grad_clip_by = 'global_norm'
        self.opt_type = 'adam'
        self.lr_schedule = None
        self.decay = 0.99
        self.momentum = 0.0
        self.epsilon = 0.1
        self.vf_loss_coeff = 0.5
        self.entropy_coeff = 0.01
        self.entropy_coeff_schedule = None
        self._separate_vf_optimizer = False
        self._lr_vf = 0.0005
        self.after_train_step = None
        self.rollout_fragment_length = 50
        self.train_batch_size = 500
        self._minibatch_size = 'auto'
        self.num_rollout_workers = 2
        self.num_gpus = 1
        self.lr = 0.0005
        self.min_time_s_per_iteration = 10
        self._tf_policy_handles_more_than_one_loss = True
        self.exploration_config = {'type': 'StochasticSampling'}
        self.num_data_loader_buffers = DEPRECATED_VALUE
        self.vtrace_drop_last_ts = DEPRECATED_VALUE

    @override(AlgorithmConfig)
    def training(self, *, vtrace: Optional[bool]=NotProvided, vtrace_clip_rho_threshold: Optional[float]=NotProvided, vtrace_clip_pg_rho_threshold: Optional[float]=NotProvided, gamma: Optional[float]=NotProvided, num_multi_gpu_tower_stacks: Optional[int]=NotProvided, minibatch_buffer_size: Optional[int]=NotProvided, minibatch_size: Optional[Union[int, str]]=NotProvided, num_sgd_iter: Optional[int]=NotProvided, replay_proportion: Optional[float]=NotProvided, replay_buffer_num_slots: Optional[int]=NotProvided, learner_queue_size: Optional[int]=NotProvided, learner_queue_timeout: Optional[float]=NotProvided, max_requests_in_flight_per_aggregator_worker: Optional[int]=NotProvided, timeout_s_sampler_manager: Optional[float]=NotProvided, timeout_s_aggregator_manager: Optional[float]=NotProvided, broadcast_interval: Optional[int]=NotProvided, num_aggregation_workers: Optional[int]=NotProvided, grad_clip: Optional[float]=NotProvided, opt_type: Optional[str]=NotProvided, lr_schedule: Optional[List[List[Union[int, float]]]]=NotProvided, decay: Optional[float]=NotProvided, momentum: Optional[float]=NotProvided, epsilon: Optional[float]=NotProvided, vf_loss_coeff: Optional[float]=NotProvided, entropy_coeff: Optional[float]=NotProvided, entropy_coeff_schedule: Optional[List[List[Union[int, float]]]]=NotProvided, _separate_vf_optimizer: Optional[bool]=NotProvided, _lr_vf: Optional[float]=NotProvided, after_train_step: Optional[Callable[[dict], None]]=NotProvided, vtrace_drop_last_ts=None, **kwargs) -> 'ImpalaConfig':
        """Sets the training related configuration.

        Args:
            vtrace: V-trace params (see vtrace_tf/torch.py).
            vtrace_clip_rho_threshold:
            vtrace_clip_pg_rho_threshold:
            gamma: Float specifying the discount factor of the Markov Decision process.
            num_multi_gpu_tower_stacks: For each stack of multi-GPU towers, how many
                slots should we reserve for parallel data loading? Set this to >1 to
                load data into GPUs in parallel. This will increase GPU memory usage
                proportionally with the number of stacks.
                Example:
                2 GPUs and `num_multi_gpu_tower_stacks=3`:
                - One tower stack consists of 2 GPUs, each with a copy of the
                model/graph.
                - Each of the stacks will create 3 slots for batch data on each of its
                GPUs, increasing memory requirements on each GPU by 3x.
                - This enables us to preload data into these stacks while another stack
                is performing gradient calculations.
            minibatch_buffer_size: How many train batches should be retained for
                minibatching. This conf only has an effect if `num_sgd_iter > 1`.
            minibatch_size: The size of minibatches that are trained over during
                each SGD iteration. If "auto", will use the same value as
                `train_batch_size`.
                Note that this setting only has an effect if
                `_enable_new_api_stack=True` and it must be a multiple of
                `rollout_fragment_length` or `sequence_length` and smaller than or equal
                to `train_batch_size`.
            num_sgd_iter: Number of passes to make over each train batch.
            replay_proportion: Set >0 to enable experience replay. Saved samples will
                be replayed with a p:1 proportion to new data samples.
            replay_buffer_num_slots: Number of sample batches to store for replay.
                The number of transitions saved total will be
                (replay_buffer_num_slots * rollout_fragment_length).
            learner_queue_size: Max queue size for train batches feeding into the
                learner.
            learner_queue_timeout: Wait for train batches to be available in minibatch
                buffer queue this many seconds. This may need to be increased e.g. when
                training with a slow environment.
            max_requests_in_flight_per_aggregator_worker: Level of queuing for replay
                aggregator operations (if using aggregator workers).
            timeout_s_sampler_manager: The timeout for waiting for sampling results
                for workers -- typically if this is too low, the manager won't be able
                to retrieve ready sampling results.
            timeout_s_aggregator_manager: The timeout for waiting for replay worker
                results -- typically if this is too low, the manager won't be able to
                retrieve ready replay requests.
            broadcast_interval: Number of training step calls before weights are
                broadcasted to rollout workers that are sampled during any iteration.
            num_aggregation_workers: Use n (`num_aggregation_workers`) extra Actors for
                multi-level aggregation of the data produced by the m RolloutWorkers
                (`num_workers`). Note that n should be much smaller than m.
                This can make sense if ingesting >2GB/s of samples, or if
                the data requires decompression.
            grad_clip: If specified, clip the global norm of gradients by this amount.
            opt_type: Either "adam" or "rmsprop".
            lr_schedule: Learning rate schedule. In the format of
                [[timestep, lr-value], [timestep, lr-value], ...]
                Intermediary timesteps will be assigned to interpolated learning rate
                values. A schedule should normally start from timestep 0.
            decay: Decay setting for the RMSProp optimizer, in case `opt_type=rmsprop`.
            momentum: Momentum setting for the RMSProp optimizer, in case
                `opt_type=rmsprop`.
            epsilon: Epsilon setting for the RMSProp optimizer, in case
                `opt_type=rmsprop`.
            vf_loss_coeff: Coefficient for the value function term in the loss function.
            entropy_coeff: Coefficient for the entropy regularizer term in the loss
                function.
            entropy_coeff_schedule: Decay schedule for the entropy regularizer.
            _separate_vf_optimizer: Set this to true to have two separate optimizers
                optimize the policy-and value networks. Only supported for some
                algorithms (APPO, IMPALA) on the old API stack.
            _lr_vf: If _separate_vf_optimizer is True, define separate learning rate
                for the value network.
            after_train_step: Callback for APPO to use to update KL, target network
                periodically. The input to the callback is the learner fetches dict.

        Returns:
            This updated AlgorithmConfig object.
        """
        if vtrace_drop_last_ts is not None:
            deprecation_warning(old='vtrace_drop_last_ts', help='The v-trace operations in RLlib have been enhanced and we are now using proper value bootstrapping at the end of each trajectory, such that no timesteps in our loss functions have to be dropped anymore.', error=True)
        super().training(**kwargs)
        if vtrace is not NotProvided:
            self.vtrace = vtrace
        if vtrace_clip_rho_threshold is not NotProvided:
            self.vtrace_clip_rho_threshold = vtrace_clip_rho_threshold
        if vtrace_clip_pg_rho_threshold is not NotProvided:
            self.vtrace_clip_pg_rho_threshold = vtrace_clip_pg_rho_threshold
        if num_multi_gpu_tower_stacks is not NotProvided:
            self.num_multi_gpu_tower_stacks = num_multi_gpu_tower_stacks
        if minibatch_buffer_size is not NotProvided:
            self.minibatch_buffer_size = minibatch_buffer_size
        if num_sgd_iter is not NotProvided:
            self.num_sgd_iter = num_sgd_iter
        if replay_proportion is not NotProvided:
            self.replay_proportion = replay_proportion
        if replay_buffer_num_slots is not NotProvided:
            self.replay_buffer_num_slots = replay_buffer_num_slots
        if learner_queue_size is not NotProvided:
            self.learner_queue_size = learner_queue_size
        if learner_queue_timeout is not NotProvided:
            self.learner_queue_timeout = learner_queue_timeout
        if broadcast_interval is not NotProvided:
            self.broadcast_interval = broadcast_interval
        if num_aggregation_workers is not NotProvided:
            self.num_aggregation_workers = num_aggregation_workers
        if max_requests_in_flight_per_aggregator_worker is not NotProvided:
            self.max_requests_in_flight_per_aggregator_worker = max_requests_in_flight_per_aggregator_worker
        if timeout_s_sampler_manager is not NotProvided:
            self.timeout_s_sampler_manager = timeout_s_sampler_manager
        if timeout_s_aggregator_manager is not NotProvided:
            self.timeout_s_aggregator_manager = timeout_s_aggregator_manager
        if grad_clip is not NotProvided:
            self.grad_clip = grad_clip
        if opt_type is not NotProvided:
            self.opt_type = opt_type
        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if decay is not NotProvided:
            self.decay = decay
        if momentum is not NotProvided:
            self.momentum = momentum
        if epsilon is not NotProvided:
            self.epsilon = epsilon
        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if entropy_coeff is not NotProvided:
            self.entropy_coeff = entropy_coeff
        if entropy_coeff_schedule is not NotProvided:
            self.entropy_coeff_schedule = entropy_coeff_schedule
        if _separate_vf_optimizer is not NotProvided:
            self._separate_vf_optimizer = _separate_vf_optimizer
        if _lr_vf is not NotProvided:
            self._lr_vf = _lr_vf
        if after_train_step is not NotProvided:
            self.after_train_step = after_train_step
        if gamma is not NotProvided:
            self.gamma = gamma
        if minibatch_size is not NotProvided:
            self._minibatch_size = minibatch_size
        return self

    @override(AlgorithmConfig)
    def validate(self) -> None:
        super().validate()
        if self.num_data_loader_buffers != DEPRECATED_VALUE:
            deprecation_warning('num_data_loader_buffers', 'num_multi_gpu_tower_stacks', error=True)
        if self._enable_new_api_stack:
            if self.entropy_coeff_schedule is not None:
                raise ValueError('`entropy_coeff_schedule` is deprecated and must be None! Use the `entropy_coeff` setting to setup a schedule.')
            Scheduler.validate(fixed_value_or_schedule=self.entropy_coeff, setting_name='entropy_coeff', description='entropy coefficient')
        elif isinstance(self.entropy_coeff, float) and self.entropy_coeff < 0.0:
            raise ValueError('`entropy_coeff` must be >= 0.0')
        if self.num_aggregation_workers > self.num_rollout_workers:
            raise ValueError('`num_aggregation_workers` must be smaller than or equal `num_rollout_workers`! Aggregation makes no sense otherwise.')
        elif self.num_aggregation_workers > self.num_rollout_workers / 2:
            logger.warning('`num_aggregation_workers` should be significantly smaller than `num_workers`! Try setting it to 0.5*`num_workers` or less.')
        if self.framework_str in ['tf', 'tf2'] and self._separate_vf_optimizer is True and (self._tf_policy_handles_more_than_one_loss is False):
            raise ValueError('`_tf_policy_handles_more_than_one_loss` must be set to True, for TFPolicy to support more than one loss term/optimizer! Try setting config.training(_tf_policy_handles_more_than_one_loss=True).')
        if self._enable_new_api_stack:
            if not (self.minibatch_size % self.rollout_fragment_length == 0 and self.minibatch_size <= self.train_batch_size):
                raise ValueError(f"`minibatch_size` ({self._minibatch_size}) must either be 'auto' or a multiple of `rollout_fragment_length` ({self.rollout_fragment_length}) while at the same time smaller than or equal to `train_batch_size` ({self.train_batch_size})!")

    @override(AlgorithmConfig)
    def get_learner_hyperparameters(self) -> ImpalaLearnerHyperparameters:
        base_hps = super().get_learner_hyperparameters()
        learner_hps = ImpalaLearnerHyperparameters(rollout_frag_or_episode_len=self.get_rollout_fragment_length(), discount_factor=self.gamma, entropy_coeff=self.entropy_coeff, vf_loss_coeff=self.vf_loss_coeff, vtrace_clip_rho_threshold=self.vtrace_clip_rho_threshold, vtrace_clip_pg_rho_threshold=self.vtrace_clip_pg_rho_threshold, **dataclasses.asdict(base_hps))
        assert (learner_hps.rollout_frag_or_episode_len is None) != (learner_hps.recurrent_seq_len is None), 'One of `rollout_frag_or_episode_len` or `recurrent_seq_len` must be not None in ImpalaLearnerHyperparameters!'
        return learner_hps

    def get_replay_ratio(self) -> float:
        """Returns replay ratio (between 0.0 and 1.0) based off self.replay_proportion.

        Formula: ratio = 1 / proportion
        """
        return 1 / self.replay_proportion if self.replay_proportion > 0 else 0.0

    @property
    def minibatch_size(self):
        return self.train_batch_size if self._minibatch_size == 'auto' else self._minibatch_size

    @override(AlgorithmConfig)
    def get_default_learner_class(self):
        if self.framework_str == 'torch':
            from ray.rllib.algorithms.impala.torch.impala_torch_learner import ImpalaTorchLearner
            return ImpalaTorchLearner
        elif self.framework_str == 'tf2':
            from ray.rllib.algorithms.impala.tf.impala_tf_learner import ImpalaTfLearner
            return ImpalaTfLearner
        else:
            raise ValueError(f"The framework {self.framework_str} is not supported. Use either 'torch' or 'tf2'.")

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> SingleAgentRLModuleSpec:
        if self.framework_str == 'tf2':
            from ray.rllib.algorithms.ppo.tf.ppo_tf_rl_module import PPOTfRLModule
            return SingleAgentRLModuleSpec(module_class=PPOTfRLModule, catalog_class=PPOCatalog)
        elif self.framework_str == 'torch':
            from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
            return SingleAgentRLModuleSpec(module_class=PPOTorchRLModule, catalog_class=PPOCatalog)
        else:
            raise ValueError(f"The framework {self.framework_str} is not supported. Use either 'torch' or 'tf2'.")