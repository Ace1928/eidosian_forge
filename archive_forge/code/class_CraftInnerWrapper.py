import os
from collections import defaultdict
import numpy as np
import minerl
import time
from typing import List
import collections
import os
import cv2
import gym
from minerl.data import DataPipeline
import sys
import time
from collections import deque, defaultdict
from enum import Enum
class CraftInnerWrapper(gym.Wrapper):
    """
    Wrapper for crafting actions
    """

    def __init__(self, env, crafts_agent):
        """
        :param env: env to wrap
        :param crafts_agent: instance of LoopCraftingAgent
        """
        super().__init__(env)
        self.crafts_agent = crafts_agent

    def step(self, action):
        """
        mix craft action with POV action
        :param action: POV action
        :return:
        """
        craft_action = self.crafts_agent.get_crafting_action()
        action = {**action, **craft_action}
        observation, reward, done, info = self.env.step(action)
        return (observation, reward, done, info)