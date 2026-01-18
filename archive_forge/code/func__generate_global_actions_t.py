from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Union
import uuid
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import MultiAgentDict
def _generate_global_actions_t(self, actions):
    global_actions_t = {agent_id: _IndexMapping() for agent_id in self._agent_ids}
    if actions:
        for t, action in enumerate(actions):
            for agent_id in action:
                global_actions_t[agent_id].append(t + self.ts_carriage_return + 1)
    return global_actions_t