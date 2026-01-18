import argparse
from pathlib import Path
import os
import ray
from ray import air, tune
from ray.rllib.examples.models.custom_loss_model import (
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LEARNER_STATS_KEY
from ray.tune.registry import get_trainable_cls
Example of using custom_loss() with an imitation learning loss under the Policy
and ModelV2 API.

The default input file is too small to learn a good policy, but you can
generate new experiences for IL training as follows:

To generate experiences:
$ ./train.py --run=PG --config='{"output": "/tmp/cartpole"}' --env=CartPole-v1

To train on experiences with joint PG + IL loss:
$ python custom_loss.py --input-files=/tmp/cartpole
