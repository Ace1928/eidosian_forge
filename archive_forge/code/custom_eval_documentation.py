import argparse
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.examples.env.simple_corridor import SimpleCorridor
from ray.rllib.utils.test_utils import check_learning_achieved
Example of a custom evaluation function.

    Args:
        algorithm: Algorithm class to evaluate.
        eval_workers: Evaluation WorkerSet.

    Returns:
        metrics: Evaluation metrics dict.
    