import gym
from minerl.env import _fake, _singleagent
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.mc import ALL_ITEMS
from typing import List
def _entry_point(self, fake: bool) -> str:
    return OBTAIN_DIAMOND_SHOVEL_ENTRY_POINT