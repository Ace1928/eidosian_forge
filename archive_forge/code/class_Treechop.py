from minerl.herobraine.env_specs.human_controls import SimpleHumanEmbodimentEnvSpec
from minerl.herobraine.hero.mc import MS_PER_STEP, STEPS_PER_MS
from minerl.herobraine.hero.handler import Handler
from typing import List
import minerl.herobraine
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.env_spec import EnvSpec
class Treechop(SimpleHumanEmbodimentEnvSpec):

    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MineRLTreechop-v0'
        super().__init__(*args, max_episode_steps=TREECHOP_LENGTH, reward_threshold=64.0, **kwargs)

    def create_rewardables(self) -> List[Handler]:
        return [handlers.RewardForCollectingItems([dict(type='log', amount=1, reward=1.0)])]

    def create_agent_start(self) -> List[Handler]:
        return super().create_agent_start() + [handlers.SimpleInventoryAgentStart([dict(type='iron_axe', quantity=1)])]

    def create_agent_handlers(self) -> List[Handler]:
        return [handlers.AgentQuitFromPossessingItem([dict(type='log', amount=64)])]

    def create_server_world_generators(self) -> List[Handler]:
        return [handlers.DefaultWorldGenerator(force_reset='true', generator_options=TREECHOP_WORLD_GENERATOR_OPTIONS)]

    def create_server_quit_producers(self) -> List[Handler]:
        return [handlers.ServerQuitFromTimeUp(TREECHOP_LENGTH * MS_PER_STEP), handlers.ServerQuitWhenAnyAgentFinishes()]

    def create_server_decorators(self) -> List[Handler]:
        return []

    def create_server_initial_conditions(self) -> List[Handler]:
        return [handlers.TimeInitialCondition(allow_passage_of_time=False), handlers.SpawningInitialCondition(allow_spawning=True)]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'survivaltreechop'

    def get_docstring(self):
        return TREECHOP_DOC