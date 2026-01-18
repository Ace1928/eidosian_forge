from collections import OrderedDict
import gymnasium as gym
from typing import Dict, List, Optional
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import AgentID
def _group_items(self, items, agg_fn=lambda gvals: list(gvals.values())):
    grouped_items = {}
    for agent_id, item in items.items():
        if agent_id in self.agent_id_to_group:
            group_id = self.agent_id_to_group[agent_id]
            if group_id in grouped_items:
                continue
            group_out = OrderedDict()
            for a in self.groups[group_id]:
                if a in items:
                    group_out[a] = items[a]
                else:
                    raise ValueError('Missing member of group {}: {}: {}'.format(group_id, a, items))
            grouped_items[group_id] = agg_fn(group_out)
        else:
            grouped_items[agent_id] = item
    return grouped_items