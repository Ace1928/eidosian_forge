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
class TrajectoryDataPipeline:
    """
    number of tools to load trajectory
    """

    @staticmethod
    def get_trajectory_names(data_dir):
        result = [os.path.basename(x) for x in DataPipeline._get_all_valid_recordings(data_dir)]
        return sorted(result)

    @staticmethod
    def map_to_dict(handler_list: list, target_space: gym.spaces.space, ignore_keys=()):

        def _map_to_dict(i: int, src: list, key: str, gym_space: gym.spaces.space, dst: dict):
            if isinstance(gym_space, gym.spaces.Dict):
                dont_count = False
                inner_dict = collections.OrderedDict()
                for idx, (k, s) in enumerate(gym_space.spaces.items()):
                    if key in ['equipped_items', 'mainhand']:
                        dont_count = True
                        i = _map_to_dict(i, src, k, s, inner_dict)
                    else:
                        _map_to_dict(idx, src[i].T, k, s, inner_dict)
                dst[key] = inner_dict
                if dont_count:
                    return i
                else:
                    return i + 1
            else:
                dst[key] = src[i]
                return i + 1
        result = collections.OrderedDict()
        index = 0
        inventory_key_list = ['coal', 'cobblestone', 'crafting_table', 'dirt', 'furnace', 'iron_axe', 'iron_ingot', 'iron_ore', 'iron_pickaxe', 'log', 'planks', 'stick', 'stone', 'stone_axe', 'stone_pickaxe', 'torch', 'wooden_axe', 'wooden_pickaxe', 'pov']
        equipped_items_key_list = ['equipped_items.mainhand.damage', 'equipped_items.mainhand.maxDamage', 'equipped_items.mainhand.type']
        key_index = 0
        for key, space in target_space.spaces.items():
            if key in ignore_keys:
                continue
            if key == 'pov':
                index = _map_to_dict(index, handler_list, key, space, result)
            if key == 'inventory':
                for inventory_key, inventory_space in space.spaces.items():
                    if key in ignore_keys:
                        continue
                    index = _map_to_dict(index, handler_list, inventory_key, inventory_space, result)
        return result

    @staticmethod
    def map_to_dict_act(handler_list: list, target_space: gym.spaces.space, ignore_keys=()):

        def _map_to_dict(i: int, src: list, key: str, gym_space: gym.spaces.space, dst: dict):
            if isinstance(gym_space, gym.spaces.Dict):
                dont_count = False
                inner_dict = collections.OrderedDict()
                for idx, (k, s) in enumerate(gym_space.spaces.items()):
                    if key in ['equipped_items', 'mainhand']:
                        dont_count = True
                        i = _map_to_dict(i, src, k, s, inner_dict)
                    else:
                        _map_to_dict(idx, src[i].T, k, s, inner_dict)
                dst[key] = inner_dict
                if dont_count:
                    return i
                else:
                    return i + 1
            else:
                dst[key] = src[i]
                return i + 1
        result = collections.OrderedDict()
        index = 0
        "\n        actions: ['action$forward', 'action$left', 'action$back', 'action$right', 'action$jump', 'action$sneak', 'action$sprint', \n        'action$attack', 'action$camera', 'action$place', 'action$equip', 'action$craft', 'action$nearbyCraft', 'action$nearbySmelt']\n\n        target_space.spaces.items(): odict_items([('attack', Discrete(2)), ('back', Discrete(2)), ('camera', Box(low=-180.0, high=180.0, shape=(2,))), \n        ('craft', Discrete(5)), ('equip', Discrete(8)), ('forward', Discrete(2)), ('jump', Discrete(2)), ('left', Discrete(2)), ('nearbyCraft', Discrete(8)), \n        ('nearbySmelt', Discrete(3)), ('place', Discrete(7)), ('right', Discrete(2)), ('sneak', Discrete(2)), ('sprint', Discrete(2))])\n        "
        key_list = ['forward', 'left', 'back', 'right', 'jump', 'sneak', 'sprint', 'attack', 'camera', 'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt']
        key_index = 0
        for key, space in target_space.spaces.items():
            key = key_list[key_index]
            key_index += 1
            if key in ignore_keys:
                continue
            index = _map_to_dict(index, handler_list, key, space, result)
        return result

    @staticmethod
    def load_video_frames(video_path, suffix_size):
        cap = cv2.VideoCapture(video_path)
        ret, frame_num = (True, 0)
        while ret:
            ret, _ = DataPipeline.read_frame(cap)
            if ret:
                frame_num += 1
        num_states = suffix_size
        frames = []
        max_frame_num = frame_num
        frame_num = 0
        cap = cv2.VideoCapture(video_path)
        for _ in range(max_frame_num - num_states):
            ret, _ = DataPipeline.read_frame(cap)
            frame_num += 1
            if not ret:
                return None
        while ret and frame_num < max_frame_num:
            ret, frame = DataPipeline.read_frame(cap)
            frames.append(frame)
            frame_num += 1
        return frames

    @classmethod
    def load_data(cls, file_dir, ignore_keys=()):
        numpy_path = str(os.path.join(file_dir, 'rendered.npz'))
        video_path = str(os.path.join(file_dir, 'recording.mp4'))
        state = np.load(numpy_path, allow_pickle=True)
        reward_vec = state['reward']
        frames = cls.load_video_frames(video_path=video_path, suffix_size=len(reward_vec) + 1)
        action_dict = collections.OrderedDict([(key, state[key]) for key in state if key.startswith('action')])
        actions = list(action_dict.keys())
        action_data = [None for _ in actions]
        for i, key in enumerate(actions):
            action_data[i] = np.asanyarray(action_dict[key])
        obs_dict = collections.OrderedDict([(key, state[key]) for key in state if key.startswith('observation$inventory$')])
        obs = list(obs_dict.keys())
        current_observation_data = [None for _ in obs]
        next_observation_data = [None for _ in obs]
        reward_vec = state['reward']
        reward_data = np.asanyarray(reward_vec, dtype=np.float32)
        done_data = [False for _ in range(len(reward_data))]
        done_data[-1] = True
        info_dict = collections.OrderedDict([(key, state[key]) for key in state if key.startswith('observation$inventory$')])
        observables = list(info_dict.keys()).copy()
        if 'pov' not in ignore_keys:
            observables.append('pov')
            current_observation_data.append(None)
            next_observation_data.append(None)
        if 'pov' not in ignore_keys:
            frames = cls.load_video_frames(video_path=video_path, suffix_size=len(reward_vec) + 1)
        else:
            frames = None
        for i, key in enumerate(observables):
            if key in ignore_keys:
                continue
            if key == 'pov':
                current_observation_data[i] = np.asanyarray(frames[:-1])
                next_observation_data[i] = np.asanyarray(frames[1:])
            else:
                current_observation_data[i] = np.asanyarray(info_dict[key][:-1])
                next_observation_data[i] = np.asanyarray(info_dict[key][1:])
        gym_spec = gym.envs.registration.spec('MineRLObtainDiamond-v0')
        observation_dict = cls.map_to_dict(current_observation_data, gym_spec._kwargs['observation_space'], ignore_keys)
        action_dict = cls.map_to_dict_act(action_data, gym_spec._kwargs['action_space'])
        next_observation_dict = cls.map_to_dict(next_observation_data, gym_spec._kwargs['observation_space'], ignore_keys)
        return [observation_dict, action_dict, reward_data, next_observation_dict, done_data]

    @classmethod
    def load_data_no_pov(cls, file_dir):
        return cls.load_data(file_dir, ignore_keys=('pov',))