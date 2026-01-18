import copy
import gymnasium as gym
import logging
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import override
from typing import Dict, Optional
from ray.rllib.examples.env.utils.interfaces import InfoAccumulationInterface
def _compute_reward(self):
    reward_red = 0.0
    reward_blue = 0.0
    generate_new_coin = False
    red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue = (False, False, False, False)
    red_first_if_both = None
    if not self.both_players_can_pick_the_same_coin:
        if self._same_pos(self.red_pos, self.coin_pos) and self._same_pos(self.blue_pos, self.coin_pos):
            red_first_if_both = bool(self.np_random.integers(low=0, high=2))
    if self.red_coin:
        if self._same_pos(self.red_pos, self.coin_pos) and (red_first_if_both is None or red_first_if_both):
            generate_new_coin = True
            reward_red += 1
            if self.asymmetric:
                reward_red += 3
            red_pick_any = True
            red_pick_red = True
        if self._same_pos(self.blue_pos, self.coin_pos) and (red_first_if_both is None or not red_first_if_both):
            generate_new_coin = True
            reward_red += -2
            reward_blue += 1
            blue_pick_any = True
    else:
        if self._same_pos(self.red_pos, self.coin_pos) and (red_first_if_both is None or red_first_if_both):
            generate_new_coin = True
            reward_red += 1
            reward_blue += -2
            if self.asymmetric:
                reward_red += 3
            red_pick_any = True
        if self._same_pos(self.blue_pos, self.coin_pos) and (red_first_if_both is None or not red_first_if_both):
            generate_new_coin = True
            reward_blue += 1
            blue_pick_blue = True
            blue_pick_any = True
    reward_list = [reward_red, reward_blue]
    if self.output_additional_info:
        self._accumulate_info(red_pick_any=red_pick_any, red_pick_red=red_pick_red, blue_pick_any=blue_pick_any, blue_pick_blue=blue_pick_blue)
    return (reward_list, generate_new_coin)