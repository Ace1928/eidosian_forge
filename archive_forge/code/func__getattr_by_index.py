from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def _getattr_by_index(self, attr: str='observations', indices: Union[int, List[int]]=-1, has_initial_value=False, global_ts: bool=True, global_ts_mapping: Optional[MultiAgentDict]=None, shift: int=0, as_list: bool=False, buffered_values: MultiAgentDict=None) -> MultiAgentDict:
    """Returns values in the form of indices: [-1, -2]."""
    if not global_ts_mapping:
        global_ts_mapping = self.global_t_to_local_t
    if global_ts:
        if isinstance(indices, list):
            indices = [self.t + idx + int(has_initial_value) if idx < 0 else idx + self.ts_carriage_return for idx in indices]
        else:
            indices = [self.t + indices + int(has_initial_value)] if indices < 0 else [indices + self.ts_carriage_return]
        if as_list:
            if buffered_values:
                return [{agent_id: (list(getattr(agent_eps, attr)) + buffered_values[agent_id])[global_ts_mapping[agent_id].find_indices([idx], shift)[0]] for agent_id, agent_eps in self.agent_episodes.items() if global_ts_mapping[agent_id].find_indices([idx], shift)} for idx in indices]
            else:
                return [{agent_id: getattr(agent_eps, attr)[global_ts_mapping[agent_id].find_indices([idx], shift)[0]] for agent_id, agent_eps in self.agent_episodes.items() if global_ts_mapping[agent_id].find_indices([idx], shift)} for idx in indices]
        elif buffered_values:
            return {agent_id: list(map((list(getattr(agent_eps, attr)) + buffered_values[agent_id]).__getitem__, global_ts_mapping[agent_id].find_indices(indices, shift))) for agent_id, agent_eps in self.agent_episodes.items() if global_ts_mapping[agent_id].find_indices(indices, shift)}
        else:
            return {agent_id: list(map(getattr(agent_eps, attr).__getitem__, global_ts_mapping[agent_id].find_indices(indices, shift))) for agent_id, agent_eps in self.agent_episodes.items() if global_ts_mapping[agent_id].find_indices(indices, shift)}
    else:
        if not isinstance(indices, list):
            indices = [indices]
        if buffered_values:
            return {agent_id: list(map((getattr(agent_eps, attr) + buffered_values[agent_id]).__getitem__, set(indices).intersection(set(range(-len(global_ts_mapping[agent_id]), len(global_ts_mapping[agent_id])))))) for agent_id, agent_eps in self.agent_episodes.items() if set(indices).intersection(set(range(-len(global_ts_mapping[agent_id]), len(global_ts_mapping[agent_id]))))}
        else:
            return {agent_id: list(map(getattr(agent_eps, attr).__getitem__, set(indices).intersection(set(range(-len(global_ts_mapping[agent_id]), len(global_ts_mapping[agent_id])))))) for agent_id, agent_eps in self.agent_episodes.items() if set(indices).intersection(set(range(-len(global_ts_mapping[agent_id]), len(global_ts_mapping[agent_id]))))}