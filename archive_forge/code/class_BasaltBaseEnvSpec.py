from typing import List, Optional, Sequence
import gym
from minerl.env import _fake, _singleagent
from minerl.herobraine import wrappers
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.env_specs import simple_embodiment
from minerl.herobraine.hero import handlers, mc
from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
class BasaltBaseEnvSpec(HumanControlEnvSpec):
    LOW_RES_SIZE = 64
    HIGH_RES_SIZE = 1024

    def __init__(self, name, demo_server_experiment_name, max_episode_steps=2400, inventory: Sequence[dict]=(), preferred_spawn_biome: str='plains'):
        self.inventory = inventory
        self.preferred_spawn_biome = preferred_spawn_biome
        self.demo_server_experiment_name = demo_server_experiment_name
        super().__init__(name=name, max_episode_steps=max_episode_steps, fov_range=[70, 70], resolution=[640, 360], gamma_range=[2, 2], guiscale_range=[1, 1], cursor_size_range=[16.0, 16.0])

    def is_from_folder(self, folder: str) -> bool:
        return folder == self.demo_server_experiment_name

    def _entry_point(self, fake: bool) -> str:
        return BASALT_GYM_ENTRY_POINT

    def create_observables(self):
        obs_handler_pov = handlers.POVObservation(self.resolution)
        return [obs_handler_pov]

    def create_agent_start(self) -> List[handlers.Handler]:
        return super().create_agent_start() + [handlers.SimpleInventoryAgentStart(self.inventory), handlers.PreferredSpawnBiome(self.preferred_spawn_biome), handlers.DoneOnDeath()]

    def create_agent_handlers(self) -> List[handlers.Handler]:
        return []

    def create_server_world_generators(self) -> List[handlers.Handler]:
        return [handlers.DefaultWorldGenerator(force_reset=True)]

    def create_server_quit_producers(self) -> List[handlers.Handler]:
        return [handlers.ServerQuitFromTimeUp(self.max_episode_steps * mc.MS_PER_STEP), handlers.ServerQuitWhenAnyAgentFinishes()]

    def create_server_decorators(self) -> List[handlers.Handler]:
        return []

    def create_server_initial_conditions(self) -> List[handlers.Handler]:
        return [handlers.TimeInitialCondition(allow_passage_of_time=False), handlers.SpawningInitialCondition(allow_spawning=True)]

    def get_blacklist_reason(self, npz_data: dict) -> Optional[str]:
        """
        Some saved demonstrations are bogus -- they only contain lobby frames.

        We can automatically skip these by checking for whether any snowballs
        were thrown.
        """
        equip = npz_data.get('observation$equipped_items$mainhand$type')
        use = npz_data.get('action$use')
        if equip is None:
            return f'Missing equip observation. Available keys: {list(npz_data.keys())}'
        if use is None:
            return f'Missing use action. Available keys: {list(npz_data.keys())}'
        assert len(equip) == len(use) + 1, (len(equip), len(use))
        for i in range(len(use)):
            if use[i] == 1 and equip[i] == 'snowball':
                return None
        return 'BasaltEnv never threw a snowball'

    def create_mission_handlers(self):
        return ()

    def create_monitors(self):
        return ()

    def create_rewardables(self):
        return ()

    def determine_success_from_rewards(self, rewards: list) -> bool:
        """Implements abstractmethod.

        Basalt environment have no rewards, so this is always False."""
        return False

    def get_docstring(self):
        return self.__class__.__doc__