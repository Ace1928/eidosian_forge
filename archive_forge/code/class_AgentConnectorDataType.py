from typing import (
import numpy as np
import gymnasium as gym
from ray.rllib.utils.annotations import ExperimentalAPI
@ExperimentalAPI
class AgentConnectorDataType:
    """Data type that is fed into and yielded from agent connectors.

    Args:
        env_id: ID of the environment.
        agent_id: ID to help identify the agent from which the data is received.
        data: A payload (``data``). With RLlib's default sampler, the payload
            is a dictionary of arbitrary data columns (obs, rewards, terminateds,
            truncateds, etc).
    """

    def __init__(self, env_id: str, agent_id: str, data: Any):
        self.env_id = env_id
        self.agent_id = agent_id
        self.data = data