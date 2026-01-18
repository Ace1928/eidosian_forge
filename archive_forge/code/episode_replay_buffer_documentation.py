from collections import deque
import copy
from typing import Any, Dict, List, Optional, Union
import numpy as np
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.base import ReplayBufferInterface
from ray.rllib.utils.typing import SampleBatchType
Returns number of timesteps that have been added in buffer's lifetime.