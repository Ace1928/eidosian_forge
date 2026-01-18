import gym
import json
import numpy as np
from copy import deepcopy
from minerl.herobraine.hero import mc, handlers
from collections import defaultdict, deque
class ReplayWrapper(gym.Wrapper):
    """
    Generic replay wrapper base class.
    Implements logic of acting with the recorded actions
    instead of policy actions for first N steps of the episode,
    where N is the number of steps in the loaded trajectory,
    abstracted away from the details of the environment.

    :param replay_file: path to the file with recorded actions. Recording
                         is assumed to be in jsonl format, where each line
                         is a json-encoded action.

    :param max_steps:   max number of steps to replay. If the recorded trajectory
                        contains more than this number of steps, the rest are
                        truncated.

    :param replay_on_reset: whether the replay is implemented in reset() method, or
                            step-by-step (in step() method via overriding ac argument
                    

    """
    IGNORE_POLICY_ACTION = 'replay_ignored_policy_action'

    def __init__(self, env, replay_file, max_steps=None, replay_on_reset=False):
        super().__init__(env)
        self.max_steps = max_steps
        self.replay_file = replay_file
        self.replay_on_reset = replay_on_reset

    def reset(self):
        self.load_actions()
        ob = self.env.reset()
        ob = self.extra_steps_on_reset(ob)
        if self.replay_on_reset:
            while len(self.actions) > 0:
                action, next_action = self.get_action_pair()
                if not self.is_on_trajectory(action):
                    break
                ac = self.replay2env(action, next_action)
                ob, _, done, _ = self.env.step(ac)
                assert not done, 'Replay put environment in done state'
        return ob

    def get_action_pair(self):
        replay_action = self.actions.popleft()
        next_action = self.actions[0] if len(self.actions) > 0 else None
        return (replay_action, next_action)

    def step(self, ac):
        ignore_ac = False
        if len(self.actions) > 0:
            replay_action, next_action = self.get_action_pair()
            if self.is_on_trajectory(replay_action):
                ac = self.replay2env(replay_action, next_action)
                ignore_ac = True
            else:
                self.actions.clear()
        ob, rew, done, info = self.env.step(ac)
        info[self.IGNORE_POLICY_ACTION] = ignore_ac
        return (ob, rew, done, info)

    def load_actions(self):
        if callable(self.replay_file):
            replay_file = self.replay_file()
        elif isinstance(self.replay_file, str):
            replay_file = self.replay_file
        else:
            raise ValueError('replay_file must be a string or a callable')
        with open(replay_file) as f:
            self.actions = deque([json.loads(l) for l in f.readlines()][:self.max_steps])

    def is_on_trajectory(self, replay_action):
        """
        Used in children to determine if the environment has not deviated
        from the recorded trajectory (otherwise, replay will be stopped)
        """
        raise NotImplementedError()

    def replay2env(self, replay_action, next_action):
        """
        Converts an action from the recording format into the environment format
        """
        raise NotImplementedError()

    def extra_steps_on_reset(self, ob):
        """
        Optional modifier for observations on reset
        Can be used to issue additional actions in case starting state is not
        perfectly recorded.
        """
        return ob