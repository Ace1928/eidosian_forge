import logging
from abc import ABC
from collections import Iterable
from typing import Dict, Optional
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.examples.env.utils.interfaces import InfoAccumulationInterface
from ray.rllib.examples.env.utils.mixins import (
class MatrixSequentialSocialDilemma(InfoAccumulationInterface, MultiAgentEnv, ABC):
    """
    A multi-agent abstract class for two player matrix games.

    PAYOUT_MATRIX: Numpy array. Along the dimension N, the action of the
    Nth player change. The last dimension is used to select the player
    whose reward you want to know.

    max_steps: number of step in one episode

    players_ids: list of the RLlib agent id of each player

    output_additional_info: ask the environment to aggregate information
    about the last episode and output them as info at the end of the
    episode.
    """

    def __init__(self, config: Optional[Dict]=None):
        if config is None:
            config = {}
        assert 'reward_randomness' not in config.keys()
        assert self.PAYOUT_MATRIX is not None
        if 'players_ids' in config:
            assert isinstance(config['players_ids'], Iterable) and len(config['players_ids']) == self.NUM_AGENTS
        self.players_ids = config.get('players_ids', ['player_row', 'player_col'])
        self.player_row_id, self.player_col_id = self.players_ids
        self.max_steps = config.get('max_steps', 20)
        self.output_additional_info = config.get('output_additional_info', True)
        self.step_count_in_current_episode = None
        if self.output_additional_info:
            self._init_info()

    def reset(self, *, seed=None, options=None):
        self.np_random, seed = seeding.np_random(seed)
        self.step_count_in_current_episode = 0
        if self.output_additional_info:
            self._reset_info()
        return ({self.player_row_id: self.NUM_STATES - 1, self.player_col_id: self.NUM_STATES - 1}, {})

    def step(self, actions: dict):
        """
        :param actions: Dict containing both actions for player_1 and player_2
        :return: observations, rewards, done, info
        """
        self.step_count_in_current_episode += 1
        action_player_row = actions[self.player_row_id]
        action_player_col = actions[self.player_col_id]
        if self.output_additional_info:
            self._accumulate_info(action_player_row, action_player_col)
        observations = self._produce_observations_invariant_to_the_player_trained(action_player_row, action_player_col)
        rewards = self._get_players_rewards(action_player_row, action_player_col)
        epi_is_done = self.step_count_in_current_episode >= self.max_steps
        if self.step_count_in_current_episode > self.max_steps:
            logger.warning('self.step_count_in_current_episode >= self.max_steps')
        info = self._get_info_for_current_epi(epi_is_done)
        return self._to_RLlib_API(observations, rewards, epi_is_done, info)

    def _produce_observations_invariant_to_the_player_trained(self, action_player_0: int, action_player_1: int):
        """
        We want to be able to use a policy trained as player 1
        for evaluation as player 2 and vice versa.
        """
        return [action_player_0 * self.NUM_ACTIONS + action_player_1, action_player_1 * self.NUM_ACTIONS + action_player_0]

    def _get_players_rewards(self, action_player_0: int, action_player_1: int):
        return [self.PAYOUT_MATRIX[action_player_0][action_player_1][0], self.PAYOUT_MATRIX[action_player_0][action_player_1][1]]

    def _to_RLlib_API(self, observations: list, rewards: list, epi_is_done: bool, info: dict):
        observations = {self.player_row_id: observations[0], self.player_col_id: observations[1]}
        rewards = {self.player_row_id: rewards[0], self.player_col_id: rewards[1]}
        if info is None:
            info = {}
        else:
            info = {self.player_row_id: info, self.player_col_id: info}
        done = {self.player_row_id: epi_is_done, self.player_col_id: epi_is_done, '__all__': epi_is_done}
        return (observations, rewards, done, done, info)

    def _get_info_for_current_epi(self, epi_is_done):
        if epi_is_done and self.output_additional_info:
            info_for_current_epi = self._get_episode_info()
        else:
            info_for_current_epi = None
        return info_for_current_epi

    def __str__(self):
        return self.NAME