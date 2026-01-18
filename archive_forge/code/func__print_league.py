import argparse
import os
import re
import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.utils import try_import_pyspiel, try_import_open_spiel
from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.examples.self_play_with_open_spiel import ask_user_for_action
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from open_spiel.python.rl_environment import Environment  # noqa: E402
def _print_league(self):
    print('--- League ---')
    print('Trainable policies (win-rates):')
    for p in sorted(self.trainable_policies):
        wr = self.win_rates[p] if p in self.win_rates else 0.0
        print(f'\t{p}: {wr}')
    print('Frozen policies:')
    for p in sorted(self.non_trainable_policies):
        wr = self.win_rates[p] if p in self.win_rates else 0.0
        print(f'\t{p}: {wr}')
    print()