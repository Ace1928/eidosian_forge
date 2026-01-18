from abc import ABC
import numpy as np
from ray.rllib.examples.env.utils.interfaces import InfoAccumulationInterface
class NPlayersNDiscreteActionsInfoMixin(InfoAccumulationInterface, ABC):
    """
    Mixin class to add logging capability in N player games with
    discrete actions.
    Logs the frequency of action profiles used
    (action profile: the set of actions used during one step by all players).
    """

    def _init_info(self):
        self.info_counters = {'n_steps_accumulated': 0}

    def _reset_info(self):
        self.info_counters = {'n_steps_accumulated': 0}

    def _get_episode_info(self):
        info = {}
        if self.info_counters['n_steps_accumulated'] > 0:
            for k, v in self.info_counters.items():
                if k != 'n_steps_accumulated':
                    info[k] = v / self.info_counters['n_steps_accumulated']
        return info

    def _accumulate_info(self, *actions):
        id = '_'.join([str(a) for a in actions])
        if id not in self.info_counters:
            self.info_counters[id] = 0
        self.info_counters[id] += 1
        self.info_counters['n_steps_accumulated'] += 1