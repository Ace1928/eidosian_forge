import argparse
import os
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls
class AssertEvalCallback(DefaultCallbacks):

    def on_train_result(self, *, algorithm, result, **kwargs):
        if 'evaluation' in result and 'hist_stats' in result['evaluation']:
            hist_stats = result['evaluation']['hist_stats']
            if algorithm.config.evaluation_duration_unit == 'episodes':
                num_episodes_done = len(hist_stats['episode_lengths'])
                if isinstance(algorithm.config.evaluation_duration, int):
                    assert num_episodes_done == algorithm.config.evaluation_duration
                else:
                    assert algorithm.config.evaluation_duration == 'auto'
                    assert num_episodes_done >= algorithm.config.evaluation_num_workers
                print(f'Number of run evaluation episodes: {num_episodes_done} (ok)!')
            else:
                num_timesteps_reported = result['evaluation']['timesteps_this_iter']
                num_timesteps_wanted = algorithm.config.evaluation_duration
                if num_timesteps_wanted != 'auto':
                    delta = num_timesteps_wanted - num_timesteps_reported
                    assert abs(delta) < 20, (delta, num_timesteps_wanted, num_timesteps_reported)
                print(f'Number of run evaluation timesteps: {num_timesteps_reported} (ok)!')
            print(f'R={result['evaluation']['episode_reward_mean']}')