import argparse
import os
import sys
import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.utils import try_import_pyspiel, try_import_open_spiel
from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.tune import CLIReporter, register_env
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.examples.rl_module.random_rl_module import RandomRLModule
from open_spiel.python.rl_environment import Environment  # noqa: E402
class SelfPlayCallback(DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self.current_opponent = 0

    def on_train_result(self, *, algorithm, result, **kwargs):
        main_rew = result['hist_stats'].pop('policy_main_reward')
        opponent_rew = list(result['hist_stats'].values())[0]
        assert len(main_rew) == len(opponent_rew)
        won = 0
        for r_main, r_opponent in zip(main_rew, opponent_rew):
            if r_main > r_opponent:
                won += 1
        win_rate = won / len(main_rew)
        result['win_rate'] = win_rate
        print(f'Iter={algorithm.iteration} win-rate={win_rate} -> ', end='')
        if win_rate > args.win_rate_threshold:
            self.current_opponent += 1
            new_pol_id = f'main_v{self.current_opponent}'
            print(f'adding new opponent to the mix ({new_pol_id}).')

            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                return 'main' if episode.episode_id % 2 == agent_id else 'main_v{}'.format(np.random.choice(list(range(1, self.current_opponent + 1))))
            main_policy = algorithm.get_policy('main')
            if algorithm.config._enable_new_api_stack:
                new_policy = algorithm.add_policy(policy_id=new_pol_id, policy_cls=type(main_policy), policy_mapping_fn=policy_mapping_fn, module_spec=SingleAgentRLModuleSpec.from_module(main_policy.model))
            else:
                new_policy = algorithm.add_policy(policy_id=new_pol_id, policy_cls=type(main_policy), policy_mapping_fn=policy_mapping_fn)
            main_state = main_policy.get_state()
            new_policy.set_state(main_state)
            algorithm.workers.sync_weights()
        else:
            print('not good enough; will keep learning ...')
        result['league_size'] = self.current_opponent + 2