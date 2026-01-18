import collections
import math
import os
import time
import observation
import networks
import gym
from gym.spaces import Dict, Discrete, Box, Tuple
import argparse
from absl import flags
from absl import logging
import grpc
import utils
import vtrace
from parametric_distribution import get_parametric_distribution_for_action_space
import tensorflow as tf
def add_batch_size(ts):
    return tf.TensorSpec([inference_batch_size] + list(ts.shape), ts.dtype, ts.name)