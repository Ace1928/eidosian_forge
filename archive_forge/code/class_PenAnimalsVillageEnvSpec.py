from typing import List, Optional, Sequence
import gym
from minerl.env import _fake, _singleagent
from minerl.herobraine import wrappers
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.env_specs import simple_embodiment
from minerl.herobraine.hero import handlers, mc
from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
class PenAnimalsVillageEnvSpec(BasaltBaseEnvSpec):
    """
.. image:: ../assets/basalt/animal_pen_village1_1-00.gif
  :scale: 100 %
  :alt:

.. image:: ../assets/basalt/animal_pen_village3_0-30.gif
  :scale: 100 %
  :alt:

.. image:: ../assets/basalt/animal_pen_village4_0-05.gif
  :scale: 100 %
  :alt:

.. image:: ../assets/basalt/animal_pen_village4_1-00.gif
  :scale: 100 %
  :alt:

After spawning in a village, build an animal pen next to one of the houses in a village.
Use your fence posts to build one animal pen that contains at least two of the same animal.
(You are only allowed to pen chickens, cows, pigs, or sheep.)
There should be at least one gate that allows players to enter and exit easily.
The animal pen should not contain more than one type of animal.
(You may kill any extra types of animals that accidentally got into the pen.)

Do not harm villagers or existing village structures in the process.

Send 1 for "ESC" key to end the episode.
"""

    def __init__(self):
        super().__init__(name='MineRLBasaltCreateVillageAnimalPen-v0', demo_server_experiment_name='village_pen_animals', max_episode_steps=5 * MINUTE, preferred_spawn_biome='plains', inventory=[dict(type='oak_fence', quantity=64), dict(type='oak_fence_gate', quantity=64), dict(type='carrot', quantity=1), dict(type='wheat_seeds', quantity=1), dict(type='wheat', quantity=1)])

    def create_agent_start(self) -> List[handlers.Handler]:
        return super().create_agent_start() + [handlers.SpawnInVillage()]