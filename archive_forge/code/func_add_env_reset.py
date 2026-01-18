from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def add_env_reset(self, *, observations: MultiAgentDict, infos: Optional[MultiAgentDict]=None, render_image: Optional[np.ndarray]=None) -> None:
    """Stores initial observation.

        Args:
            observations: A dictionary mapping agent ids to initial observations.
                Note that some agents may not have an initial observation.
            infos: A dictionary mapping agent ids to initial info dicts.
                Note that some agents may not have an initial info dict.
            render_image: An RGB uint8 image from rendering the environment.
        """
    assert not self.is_done
    assert self.t == self.t_started == 0
    infos = infos or {}
    if len(self.global_t_to_local_t) == 0:
        self.global_t_to_local_t = {agent_id: _IndexMapping() for agent_id in self._agent_ids}
    if render_image is not None:
        self.render_images.append(render_image)
    for agent_id in observations.keys():
        self.global_t_to_local_t[agent_id].append(self.t)
        self.agent_episodes[agent_id].add_env_reset(observation=observations[agent_id], infos=infos.get(agent_id))