import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EpisodeID, PolicyID, TensorType
@abstractmethod
def add_action_reward_next_obs(self, episode_id: EpisodeID, agent_id: AgentID, env_id: EnvID, policy_id: PolicyID, agent_done: bool, values: Dict[str, TensorType]) -> None:
    """Add the given dictionary (row) of values to this collector.

        The incoming data (`values`) must include action, reward, terminated, truncated,
        and next_obs information and may include any other information.
        For the initial observation (after Env.reset()) of the given agent/
        episode-ID combination, `add_initial_obs()` must be called instead.

        Args:
            episode_id: Unique id for the episode we are adding
                values for.
            agent_id: Unique id for the agent we are adding
                values for.
            env_id: The environment index (in a vectorized setup).
            policy_id: Unique id for policy controlling the agent.
            agent_done: Whether the given agent is done (terminated or truncated) with
                its trajectory (the multi-agent episode may still be ongoing).
            values (Dict[str, TensorType]): Row of values to add for this
                agent. This row must contain the keys SampleBatch.ACTION,
                REWARD, NEW_OBS, TERMINATED, and TRUNCATED.

        .. testcode::
            :skipif: True

            obs, info = env.reset()
            collector.add_init_obs(12345, 0, "pol0", obs)
            obs, r, terminated, truncated, info = env.step(action)
            collector.add_action_reward_next_obs(
                12345,
                0,
                "pol0",
                agent_done=False,
                values={
                    "action": action, "obs": obs, "reward": r,
                    "terminated": terminated, "truncated": truncated
                },
            )
        """
    raise NotImplementedError