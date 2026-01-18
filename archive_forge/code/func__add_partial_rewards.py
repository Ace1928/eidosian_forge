from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def _add_partial_rewards(self, successor):
    for agent_id, agent_partial_rewards in self.partial_rewards.items():
        if self.global_t_to_local_t[agent_id]:
            indices_to_keep = self.partial_rewards_t[agent_id].find_indices_right(self.global_t_to_local_t[agent_id][-1])
        elif self.partial_rewards_t[agent_id]:
            indices_to_keep = list(range(len(self.partial_rewards_t[agent_id])))
        else:
            indices_to_keep = []
        successor.partial_rewards_t[agent_id] = _IndexMapping(map(self.partial_rewards_t[agent_id].__getitem__, indices_to_keep))
        successor.partial_rewards[agent_id] = list(map(agent_partial_rewards.__getitem__, indices_to_keep))
        if not self.agent_episodes[agent_id].is_done:
            successor.agent_buffers[agent_id]['rewards'].get_nowait()
            successor.agent_buffers[agent_id]['rewards'].put_nowait(sum(successor.partial_rewards[agent_id]))
    return successor