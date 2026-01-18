import copy
from collections import Iterable
import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame
from ray.rllib.utils import override
class VectorizedCoinGame(CoinGame):
    """
    Vectorized Coin Game environment.
    """

    def __init__(self, config=None):
        if config is None:
            config = {}
        super().__init__(config)
        self.batch_size = config.get('batch_size', 1)
        self.force_vectorized = config.get('force_vectorize', False)
        assert self.grid_size == 3, 'hardcoded in the generate_state function'

    @override(CoinGame)
    def _randomize_color_and_player_positions(self):
        self.red_coin = np.random.randint(2, size=self.batch_size)
        self.red_pos = np.random.randint(self.grid_size, size=(self.batch_size, 2))
        self.blue_pos = np.random.randint(self.grid_size, size=(self.batch_size, 2))
        self.coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)
        self._players_do_not_overlap_at_start()

    @override(CoinGame)
    def _players_do_not_overlap_at_start(self):
        for i in range(self.batch_size):
            while _same_pos(self.red_pos[i], self.blue_pos[i]):
                self.blue_pos[i] = np.random.randint(self.grid_size, size=2)

    @override(CoinGame)
    def _generate_coin(self):
        generate = np.ones(self.batch_size, dtype=bool)
        self.coin_pos = generate_coin(self.batch_size, generate, self.red_coin, self.red_pos, self.blue_pos, self.coin_pos, self.grid_size)

    @override(CoinGame)
    def _generate_observation(self):
        obs = generate_observations_wt_numba_optimization(self.batch_size, self.red_pos, self.blue_pos, self.coin_pos, self.red_coin, self.grid_size)
        obs = self._get_obs_invariant_to_the_player_trained(obs)
        obs, _ = self._optional_unvectorize(obs)
        return obs

    def _optional_unvectorize(self, obs, rewards=None):
        if self.batch_size == 1 and (not self.force_vectorized):
            obs = [one_obs[0, ...] for one_obs in obs]
            if rewards is not None:
                rewards[0], rewards[1] = (rewards[0][0], rewards[1][0])
        return (obs, rewards)

    @override(CoinGame)
    def step(self, actions: Iterable):
        actions = self._from_RLlib_API_to_list(actions)
        self.step_count_in_current_episode += 1
        self.red_pos, self.blue_pos, rewards, self.coin_pos, observation, self.red_coin, red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue = vectorized_step_wt_numba_optimization(actions, self.batch_size, self.red_pos, self.blue_pos, self.coin_pos, self.red_coin, self.grid_size, self.asymmetric, self.max_steps, self.both_players_can_pick_the_same_coin)
        if self.output_additional_info:
            self._accumulate_info(red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue)
        obs = self._get_obs_invariant_to_the_player_trained(observation)
        obs, rewards = self._optional_unvectorize(obs, rewards)
        return self._to_RLlib_API(obs, rewards)

    @override(CoinGame)
    def _get_episode_info(self):
        player_red_info, player_blue_info = ({}, {})
        if len(self.red_pick) > 0:
            red_pick = sum(self.red_pick)
            player_red_info['pick_speed'] = red_pick / (len(self.red_pick) * self.batch_size)
            if red_pick > 0:
                player_red_info['pick_own_color'] = sum(self.red_pick_own) / red_pick
        if len(self.blue_pick) > 0:
            blue_pick = sum(self.blue_pick)
            player_blue_info['pick_speed'] = blue_pick / (len(self.blue_pick) * self.batch_size)
            if blue_pick > 0:
                player_blue_info['pick_own_color'] = sum(self.blue_pick_own) / blue_pick
        return (player_red_info, player_blue_info)

    @override(CoinGame)
    def _from_RLlib_API_to_list(self, actions):
        ac_red = actions[self.player_red_id]
        ac_blue = actions[self.player_blue_id]
        if not isinstance(ac_red, Iterable):
            assert not isinstance(ac_blue, Iterable)
            ac_red, ac_blue = ([ac_red], [ac_blue])
        actions = [ac_red, ac_blue]
        actions = np.array(actions).T
        return actions

    def _save_env(self):
        env_save_state = {'red_pos': self.red_pos, 'blue_pos': self.blue_pos, 'coin_pos': self.coin_pos, 'red_coin': self.red_coin, 'grid_size': self.grid_size, 'asymmetric': self.asymmetric, 'batch_size': self.batch_size, 'step_count_in_current_episode': self.step_count_in_current_episode, 'max_steps': self.max_steps, 'red_pick': self.red_pick, 'red_pick_own': self.red_pick_own, 'blue_pick': self.blue_pick, 'blue_pick_own': self.blue_pick_own, 'both_players_can_pick_the_same_coin': self.both_players_can_pick_the_same_coin}
        return copy.deepcopy(env_save_state)

    def _load_env(self, env_state):
        for k, v in env_state.items():
            self.__setattr__(k, v)