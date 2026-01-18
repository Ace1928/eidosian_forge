from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
class _IndexMapping(list):
    """Provides lists with a method to find multiple elements.

    This class is used for the timestep mapping which is central to
    the multi-agent episode. For each agent the timestep mapping is
    implemented with an `IndexMapping`.

    The `IndexMapping.find_indices` method simplifies the search for
    multiple environment timesteps at which some agents have stepped.
    See for example `MultiAgentEpisode.get_observations()`.
    """

    def find_indices(self, indices_to_find: List[int], shift: int=0):
        """Returns global timesteps at which an agent stepped.

        The function returns for a given list of indices the ones
        that are stored in the `IndexMapping`.

        Args:
            indices_to_find: A list of indices that should be
                found in the `IndexMapping`.

        Returns:
            A list of indices at which to find the `indices_to_find`
            in `self`. This could be empty if none of the given
            indices are in `IndexMapping`.
        """
        indices = []
        for num in indices_to_find:
            if num in self and self.index(num) + shift >= 0:
                indices.append(self.index(num) + shift)
        return indices

    def find_indices_right(self, threshold: int, shift: bool=0):
        indices = []
        for num in reversed(self):
            if num <= threshold:
                break
            elif self.index(num) + shift < 0:
                continue
            else:
                indices.append(max(self.index(num) + shift, 0))
        return list(reversed(indices))

    def find_indices_right_equal(self, threshold: int, shift: bool=0):
        indices = []
        for num in reversed(self):
            if num < threshold:
                break
            elif self.index(num) + shift < 0:
                continue
            else:
                indices.append(max(self.index(num) + shift, 0))
        return list(reversed(indices))

    def find_indices_left_equal(self, threshold: int, shift: bool=0):
        indices = []
        for num in self:
            if num > threshold or self.index(num) + shift < 0:
                break
            else:
                indices.append(self.index(num))
        return indices

    def find_indices_between_right_equal(self, threshold_left: int, threshold_right: int, shift: int=0):
        indices = []
        for num in self:
            if num > threshold_right:
                break
            elif num <= threshold_left or self.index(num) + shift < 0:
                continue
            else:
                indices.append(self.index(num) + shift)
        return indices