from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
from typing import List
import gym
import minerl.herobraine
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.hero.handlers import TranslationHandler
from minerl.herobraine.hero.mc import ALL_ITEMS, INVERSE_KEYMAP
from minerl.herobraine.env_spec import EnvSpec
from collections import OrderedDict
class Survival(SimpleEmbodimentEnvSpec):

    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MineRLSurvival-v0'
        if 'max_episode_steps' not in kwargs:
            kwargs['max_episode_steps'] = 24 * 60 * 60 * 20
        self.episode_len = kwargs['max_episode_steps']
        super().__init__(*args, **kwargs)

    def create_rewardables(self) -> List[Handler]:
        return []

    def create_agent_start(self) -> List[Handler]:
        return []

    def create_agent_handlers(self) -> List[Handler]:
        return []

    def create_server_world_generators(self) -> List[Handler]:
        return [handlers.DefaultWorldGenerator(force_reset='true', generator_options='')]

    def create_server_quit_producers(self) -> List[Handler]:
        return [handlers.ServerQuitFromTimeUp(self.episode_len * MS_PER_STEP), handlers.ServerQuitWhenAnyAgentFinishes()]

    def create_server_decorators(self) -> List[Handler]:
        return []

    def create_server_initial_conditions(self) -> List[Handler]:
        return [handlers.TimeInitialCondition(allow_passage_of_time=True, start_time=0), handlers.SpawningInitialCondition(allow_spawning=True)]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return True

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'none'

    def get_docstring(self):
        return SURVIVIAL_DOC

    def create_mission_handlers(self) -> List[Handler]:
        return []

    def create_observables(self) -> List[Handler]:
        return [handlers.POVObservation(self.resolution), handlers.FlatInventoryObservation(ALL_ITEMS), handlers.TypeObservation('mainhand', none + ALL_ITEMS + other), handlers.DamageObservation('mainhand'), handlers.MaxDamageObservation('mainhand'), handlers.ObservationFromCurrentLocation()]

    def create_actionables(self) -> List[Handler]:
        actionables = [handlers.KeyboardAction(k, v) for k, v in INVERSE_KEYMAP.items()]
        actionables += [handlers.CraftItem(none + ALL_ITEMS), handlers.CraftItemNearby(none + ALL_ITEMS), handlers.SmeltItemNearby(none + ALL_ITEMS), handlers.PlaceBlock(none + ALL_ITEMS), handlers.EquipItem(none + ALL_ITEMS), handlers.Camera()]
        return actionables