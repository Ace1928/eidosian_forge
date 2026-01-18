from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def accumulate_partial_rewards(self, agent_id: Union[str, int], indices: List[int], return_indices: bool=False):
    """Accumulates rewards along the interval between two indices.

        Assumes the indices are sorted ascendingly. Opeartes along the
        half-open interval (last_index, index].

        Args:
            agent_id: Either string or integer. The unique id of the agent in
                the `MultiAgentEpisode`.
            indices: List of integers. The ascendingly sorted indices for which
                the rewards should be accumulated.

        Returns: A list of accumulated rewards for the indices `1:len(indices)`.
        """
    if return_indices:
        index_interval_map = {indices[indices.index(idx) + 1]: self.partial_rewards_t[agent_id].find_indices_between_right_equal(idx, indices[indices.index(idx) + 1]) for idx in indices[:-1] if self.partial_rewards_t[agent_id].find_indices_between_right_equal(idx, indices[indices.index(idx) + 1])}
        return ([sum(map(self.partial_rewards[agent_id].__getitem__, v)) for v in index_interval_map.values()], list(index_interval_map.keys()))
    else:
        return [sum([self.partial_rewards[agent_id][i] for i in self.partial_rewards_t[agent_id].find_indices_between_right_equal(idx, indices[indices.index(idx) + 1])]) for idx in indices[:-1] if self.partial_rewards_t[agent_id].find_indices_between_right_equal(idx, indices[indices.index(idx) + 1])]