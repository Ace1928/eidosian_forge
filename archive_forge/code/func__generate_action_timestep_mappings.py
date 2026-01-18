from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def _generate_action_timestep_mappings(self):
    return {agent_id: _IndexMapping([self.global_actions_t[agent_id][-1]]) if agent_buffer['actions'].full() else _IndexMapping() for agent_id, agent_buffer in self.agent_buffers.items()}