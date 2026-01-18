import logging
from typing import List, Optional, Type, Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.simple_q.simple_q_tf_policy import (
from ray.rllib.algorithms.simple_q.simple_q_torch_policy import SimpleQTorchPolicy
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.metrics import (
from ray.rllib.utils.replay_buffers.utils import (
from ray.rllib.utils.typing import ResultDict
class SimpleQConfig(AlgorithmConfig):
    """Defines a configuration class from which a SimpleQ Algorithm can be built.

    Example:
        >>> from ray.rllib.algorithms.simple_q import SimpleQConfig
        >>> config = SimpleQConfig()
        >>> print(config.replay_buffer_config)  # doctest: +SKIP
        >>> replay_config = config.replay_buffer_config.update(
        >>>     {
        >>>         "capacity":  40000,
        >>>     }
        >>> )
        >>> config.training(replay_buffer_config=replay_config)
        ...       .resources(num_gpus=1)
        ...       .rollouts(num_rollout_workers=3)

    Example:
        >>> from ray.rllib.algorithms.simple_q import SimpleQConfig
        >>> from ray import air
        >>> from ray import tune
        >>> config = SimpleQConfig()
        >>> config.training(adam_epsilon=tune.grid_search([1e-8, 5e-8, 1e-7])
        >>> config.environment(env="CartPole-v1")
        >>> tune.Tuner(  # doctest: +SKIP
        ...     "SimpleQ",
        ...     run_config=air.RunConfig(stop={"episode_reward_mean": 200}),
        ...     param_space=config.to_dict()
        ... ).fit()

    Example:
        >>> from ray.rllib.algorithms.simple_q import SimpleQConfig
        >>> config = SimpleQConfig()
        >>> print(config.exploration_config)  # doctest: +SKIP
        >>> explore_config = config.exploration_config.update(
        >>>     {
        >>>         "initial_epsilon": 1.5,
        >>>         "final_epsilon": 0.01,
        >>>         "epsilon_timesteps": 5000,
        >>>     })
        >>> config = SimpleQConfig().rollouts(rollout_fragment_length=32)
        >>>                         .exploration(exploration_config=explore_config)

    Example:
        >>> from ray.rllib.algorithms.simple_q import SimpleQConfig
        >>> config = SimpleQConfig()
        >>> print(config.exploration_config)  # doctest: +SKIP
        >>> explore_config = config.exploration_config.update(
        >>>     {
        >>>         "type": "softq",
        >>>         "temperature": [1.0],
        >>>     })
        >>> config = SimpleQConfig().training(lr_schedule=[[1, 1e-3], [500, 5e-3]])        >>>                         .exploration(exploration_config=explore_config)
    """

    def __init__(self, algo_class=None):
        """Initializes a SimpleQConfig instance."""
        super().__init__(algo_class=algo_class or SimpleQ)
        self.target_network_update_freq = 500
        self.replay_buffer_config = {'type': 'MultiAgentReplayBuffer', 'capacity': 50000, 'replay_sequence_length': 1}
        self.num_steps_sampled_before_learning_starts = 1000
        self.store_buffer_in_checkpoints = False
        self.lr_schedule = None
        self.adam_epsilon = 1e-08
        self.grad_clip = 40.0
        self.grad_clip_by = 'global_norm'
        self.tau = 1.0
        self.rollout_fragment_length = 4
        self.lr = 0.0005
        self.train_batch_size = 32
        self.exploration_config = {'type': 'EpsilonGreedy', 'initial_epsilon': 1.0, 'final_epsilon': 0.02, 'epsilon_timesteps': 10000}
        self.evaluation(evaluation_config=AlgorithmConfig.overrides(explore=False))
        self.min_time_s_per_iteration = None
        self.min_sample_timesteps_per_iteration = 1000
        self.buffer_size = DEPRECATED_VALUE
        self.prioritized_replay = DEPRECATED_VALUE
        self.learning_starts = DEPRECATED_VALUE
        self.replay_batch_size = DEPRECATED_VALUE
        self.replay_sequence_length = None
        self.prioritized_replay_alpha = DEPRECATED_VALUE
        self.prioritized_replay_beta = DEPRECATED_VALUE
        self.prioritized_replay_eps = DEPRECATED_VALUE

    @override(AlgorithmConfig)
    def training(self, *, target_network_update_freq: Optional[int]=NotProvided, replay_buffer_config: Optional[dict]=NotProvided, store_buffer_in_checkpoints: Optional[bool]=NotProvided, lr_schedule: Optional[List[List[Union[int, float]]]]=NotProvided, adam_epsilon: Optional[float]=NotProvided, grad_clip: Optional[int]=NotProvided, num_steps_sampled_before_learning_starts: Optional[int]=NotProvided, tau: Optional[float]=NotProvided, **kwargs) -> 'SimpleQConfig':
        """Sets the training related configuration.

        Args:
            target_network_update_freq: Update the target network every
                `target_network_update_freq` sample steps.
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
            store_buffer_in_checkpoints: Set this to True, if you want the contents of
                your buffer(s) to be stored in any saved checkpoints as well.
                Warnings will be created if:
                - This is True AND restoring from a checkpoint that contains no buffer
                data.
                - This is False AND restoring from a checkpoint that does contain
                buffer data.
            lr_schedule: Learning rate schedule. In the format of [[timestep, value],
                [timestep, value], ...]. A schedule should normally start from
                timestep 0.
            adam_epsilon: Adam optimizer's epsilon hyper parameter.
            grad_clip: If not None, clip gradients during optimization at this value.
            num_steps_sampled_before_learning_starts: Number of timesteps to collect
                from rollout workers before we start sampling from replay buffers for
                learning. Whether we count this in agent steps  or environment steps
                depends on config.multi_agent(count_steps_by=..).
            tau: Update the target by 	au * policy + (1-	au) * target_policy.

        Returns:
            This updated AlgorithmConfig object.
        """
        super().training(**kwargs)
        if target_network_update_freq is not NotProvided:
            self.target_network_update_freq = target_network_update_freq
        if replay_buffer_config is not NotProvided:
            new_replay_buffer_config = deep_update({'replay_buffer_config': self.replay_buffer_config}, {'replay_buffer_config': replay_buffer_config}, False, ['replay_buffer_config'], ['replay_buffer_config'])
            self.replay_buffer_config = new_replay_buffer_config['replay_buffer_config']
        if store_buffer_in_checkpoints is not NotProvided:
            self.store_buffer_in_checkpoints = store_buffer_in_checkpoints
        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if adam_epsilon is not NotProvided:
            self.adam_epsilon = adam_epsilon
        if grad_clip is not NotProvided:
            self.grad_clip = grad_clip
        if num_steps_sampled_before_learning_starts is not NotProvided:
            self.num_steps_sampled_before_learning_starts = num_steps_sampled_before_learning_starts
        if tau is not NotProvided:
            self.tau = tau
        return self

    @override(AlgorithmConfig)
    def validate(self) -> None:
        super().validate()
        if self.exploration_config['type'] == 'ParameterNoise':
            if self.batch_mode != 'complete_episodes':
                raise ValueError("ParameterNoise Exploration requires `batch_mode` to be 'complete_episodes'. Try setting `config.rollouts(batch_mode='complete_episodes')`.")
        if not self.in_evaluation:
            validate_buffer_config(self)