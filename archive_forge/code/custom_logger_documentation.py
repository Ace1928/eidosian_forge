import argparse
import os
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import Logger, LegacyLoggerCallback
from ray.tune.registry import get_trainable_cls
Logs results by simply printing out everything.