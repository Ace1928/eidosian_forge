import gym
from minerl.env import _fake, _singleagent
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.mc import ALL_ITEMS
from typing import List
class ObtainDiamondShovelWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.rewarded_items = DIAMOND_ITEMS
        self.seen = [0] * len(self.rewarded_items)
        self.timeout = self.env.task.max_episode_steps
        self.num_steps = 0
        self.episode_over = False

    def step(self, action: dict):
        if self.episode_over:
            raise RuntimeError('Expected `reset` after episode terminated, not `step`.')
        observation, reward, done, info = super().step(action)
        for i, [item_list, rew] in enumerate(self.rewarded_items):
            if not self.seen[i]:
                for item in item_list:
                    if observation['inventory'][item] > 0:
                        if i == len(self.rewarded_items) - 1:
                            done = True
                        reward += rew
                        self.seen[i] = 1
                        break
        self.num_steps += 1
        if self.num_steps >= self.timeout:
            done = True
        self.episode_over = done
        return (observation, reward, done, info)

    def reset(self):
        self.seen = [0] * len(self.rewarded_items)
        self.episode_over = False
        obs = super().reset()
        return obs