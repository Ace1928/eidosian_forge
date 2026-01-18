from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ddpg.ddpg import DDPG, DDPGConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
class TD3Config(DDPGConfig):

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or TD3)
        self.twin_q = True
        self.policy_delay = 2
        self.smooth_target_policy = (True,)
        self.l2_reg = 0.0
        self.tau = 0.005
        self.train_batch_size = 100
        self.replay_buffer_config = {'type': 'MultiAgentReplayBuffer', 'prioritized_replay': DEPRECATED_VALUE, 'capacity': 1000000, 'worker_side_prioritization': False}
        self.num_steps_sampled_before_learning_starts = 10000
        self.exploration_config = {'type': 'GaussianNoise', 'random_timesteps': 10000, 'stddev': 0.1, 'initial_scale': 1.0, 'final_scale': 1.0, 'scale_timesteps': 1}