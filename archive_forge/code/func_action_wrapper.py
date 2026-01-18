import gym
import itertools
import minerl
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tqdm
import baselines.common.tf_util as U
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule
import logging
import coloredlogs
def action_wrapper(action_int):
    act = {'forward': 1, 'back': 0, 'left': 0, 'right': 0, 'jump': 0, 'sneak': 0, 'sprint': 0, 'attack': 1, 'camera': [0, 0]}
    if action_int == 0:
        act['jump'] = 1
    elif action_int == 1:
        act['camera'] = [0, 10]
    elif action_int == 2:
        act['camera'] = [0, -10]
    elif action_int == 3:
        act['forward'] = 0
    return act.copy()