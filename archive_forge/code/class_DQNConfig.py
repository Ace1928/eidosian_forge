import logging
from typing import List, Optional, Type, Callable
import numpy as np
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.simple_q.simple_q import (
from ray.rllib.execution.rollout_ops import (
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.execution.train_ops import (
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.utils import update_priorities_in_replay_buffer
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.metrics import (
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.replay_buffers.utils import sample_min_n_steps_from_buffer
class DQNConfig(SimpleQConfig):
    """Defines a configuration class from which a DQN Algorithm can be built.

    .. testcode::

        from ray.rllib.algorithms.dqn.dqn import DQNConfig
        config = DQNConfig()

        replay_config = {
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 60000,
                "prioritized_replay_alpha": 0.5,
                "prioritized_replay_beta": 0.5,
                "prioritized_replay_eps": 3e-6,
            }

        config = config.training(replay_buffer_config=replay_config)
        config = config.resources(num_gpus=0)
        config = config.rollouts(num_rollout_workers=1)
        config = config.environment("CartPole-v1")
        algo = DQN(config=config)
        algo.train()
        del algo

    .. testcode::

        from ray.rllib.algorithms.dqn.dqn import DQNConfig
        from ray import air
        from ray import tune
        config = DQNConfig()
        config = config.training(
            num_atoms=tune.grid_search([1,]))
        config = config.environment(env="CartPole-v1")
        tune.Tuner(
            "DQN",
            run_config=air.RunConfig(stop={"training_iteration":1}),
            param_space=config.to_dict()
        ).fit()

    .. testoutput::
        :hide:

        ...


    """

    def __init__(self, algo_class=None):
        """Initializes a DQNConfig instance."""
        super().__init__(algo_class=algo_class or DQN)
        self.num_atoms = 1
        self.v_min = -10.0
        self.v_max = 10.0
        self.noisy = False
        self.sigma0 = 0.5
        self.dueling = True
        self.hiddens = [256]
        self.double_q = True
        self.n_step = 1
        self.before_learn_on_batch = None
        self.training_intensity = None
        self.td_error_loss_fn = 'huber'
        self.categorical_distribution_temperature = 1.0
        self.replay_buffer_config = {'type': 'MultiAgentPrioritizedReplayBuffer', 'prioritized_replay': DEPRECATED_VALUE, 'capacity': 50000, 'prioritized_replay_alpha': 0.6, 'prioritized_replay_beta': 0.4, 'prioritized_replay_eps': 1e-06, 'replay_sequence_length': 1, 'worker_side_prioritization': False}
        self.rollout_fragment_length = 'auto'

    @override(SimpleQConfig)
    def training(self, *, num_atoms: Optional[int]=NotProvided, v_min: Optional[float]=NotProvided, v_max: Optional[float]=NotProvided, noisy: Optional[bool]=NotProvided, sigma0: Optional[float]=NotProvided, dueling: Optional[bool]=NotProvided, hiddens: Optional[int]=NotProvided, double_q: Optional[bool]=NotProvided, n_step: Optional[int]=NotProvided, before_learn_on_batch: Callable[[Type[MultiAgentBatch], List[Type[Policy]], Type[int]], Type[MultiAgentBatch]]=NotProvided, training_intensity: Optional[float]=NotProvided, td_error_loss_fn: Optional[str]=NotProvided, categorical_distribution_temperature: Optional[float]=NotProvided, **kwargs) -> 'DQNConfig':
        """Sets the training related configuration.

        Args:
            num_atoms: Number of atoms for representing the distribution of return.
                When this is greater than 1, distributional Q-learning is used.
            v_min: Minimum value estimation
            v_max: Maximum value estimation
            noisy: Whether to use noisy network to aid exploration. This adds parametric
                noise to the model weights.
            sigma0: Control the initial parameter noise for noisy nets.
            dueling: Whether to use dueling DQN.
            hiddens: Dense-layer setup for each the advantage branch and the value
                branch
            double_q: Whether to use double DQN.
            n_step: N-step for Q-learning.
            before_learn_on_batch: Callback to run before learning on a multi-agent
                batch of experiences.
            training_intensity: The intensity with which to update the model (vs
                collecting samples from the env).
                If None, uses "natural" values of:
                `train_batch_size` / (`rollout_fragment_length` x `num_workers` x
                `num_envs_per_worker`).
                If not None, will make sure that the ratio between timesteps inserted
                into and sampled from the buffer matches the given values.
                Example:
                training_intensity=1000.0
                train_batch_size=250
                rollout_fragment_length=1
                num_workers=1 (or 0)
                num_envs_per_worker=1
                -> natural value = 250 / 1 = 250.0
                -> will make sure that replay+train op will be executed 4x asoften as
                rollout+insert op (4 * 250 = 1000).
                See: rllib/algorithms/dqn/dqn.py::calculate_rr_weights for further
                details.
            replay_buffer_config: Replay buffer config.
                Examples:
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentReplayBuffer",
                "capacity": 50000,
                "replay_sequence_length": 1,
                }
                - OR -
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 50000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
                "replay_sequence_length": 1,
                }
                - Where -
                prioritized_replay_alpha: Alpha parameter controls the degree of
                prioritization in the buffer. In other words, when a buffer sample has
                a higher temporal-difference error, with how much more probability
                should it drawn to use to update the parametrized Q-network. 0.0
                corresponds to uniform probability. Setting much above 1.0 may quickly
                result as the sampling distribution could become heavily “pointy” with
                low entropy.
                prioritized_replay_beta: Beta parameter controls the degree of
                importance sampling which suppresses the influence of gradient updates
                from samples that have higher probability of being sampled via alpha
                parameter and the temporal-difference error.
                prioritized_replay_eps: Epsilon parameter sets the baseline probability
                for sampling so that when the temporal-difference error of a sample is
                zero, there is still a chance of drawing the sample.
            td_error_loss_fn: "huber" or "mse". loss function for calculating TD error
                when num_atoms is 1. Note that if num_atoms is > 1, this parameter
                is simply ignored, and softmax cross entropy loss will be used.
            categorical_distribution_temperature: Set the temperature parameter used
                by Categorical action distribution. A valid temperature is in the range
                of [0, 1]. Note that this mostly affects evaluation since TD error uses
                argmax for return calculation.

        Returns:
            This updated AlgorithmConfig object.
        """
        super().training(**kwargs)
        if num_atoms is not NotProvided:
            self.num_atoms = num_atoms
        if v_min is not NotProvided:
            self.v_min = v_min
        if v_max is not NotProvided:
            self.v_max = v_max
        if noisy is not NotProvided:
            self.noisy = noisy
        if sigma0 is not NotProvided:
            self.sigma0 = sigma0
        if dueling is not NotProvided:
            self.dueling = dueling
        if hiddens is not NotProvided:
            self.hiddens = hiddens
        if double_q is not NotProvided:
            self.double_q = double_q
        if n_step is not NotProvided:
            self.n_step = n_step
        if before_learn_on_batch is not NotProvided:
            self.before_learn_on_batch = before_learn_on_batch
        if training_intensity is not NotProvided:
            self.training_intensity = training_intensity
        if td_error_loss_fn is not NotProvided:
            self.td_error_loss_fn = td_error_loss_fn
        if categorical_distribution_temperature is not NotProvided:
            self.categorical_distribution_temperature = categorical_distribution_temperature
        return self

    @override(SimpleQConfig)
    def validate(self) -> None:
        super().validate()
        if self.td_error_loss_fn not in ['huber', 'mse']:
            raise ValueError("`td_error_loss_fn` must be 'huber' or 'mse'!")
        if not self.in_evaluation and self.rollout_fragment_length != 'auto' and (self.rollout_fragment_length < self.n_step):
            raise ValueError(f'Your `rollout_fragment_length` ({self.rollout_fragment_length}) is smaller than `n_step` ({self.n_step})! Try setting config.rollouts(rollout_fragment_length={self.n_step}).')
        if self.exploration_config['type'] == 'ParameterNoise':
            if self.batch_mode != 'complete_episodes':
                raise ValueError("ParameterNoise Exploration requires `batch_mode` to be 'complete_episodes'. Try setting `config.rollouts(batch_mode='complete_episodes')`.")
            if self.noisy:
                raise ValueError('ParameterNoise Exploration and `noisy` network cannot be used at the same time!')

    @override(AlgorithmConfig)
    def get_rollout_fragment_length(self, worker_index: int=0) -> int:
        if self.rollout_fragment_length == 'auto':
            return self.n_step
        else:
            return self.rollout_fragment_length