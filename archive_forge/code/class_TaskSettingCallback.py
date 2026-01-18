import argparse
import gymnasium as gym
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls
class TaskSettingCallback(DefaultCallbacks):
    """Custom callback to verify, we can set the task on each remote sub-env."""

    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        """Curriculum learning as seen in Ray docs"""
        if result['episode_reward_mean'] > 0.0:
            phase = 0
        else:
            phase = 1
        algorithm.workers.foreach_env(lambda env: env.set_task.remote(phase))