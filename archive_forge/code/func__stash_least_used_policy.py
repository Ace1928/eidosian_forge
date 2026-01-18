from collections import deque
import threading
from typing import Dict, Set
import logging
import ray
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.typing import PolicyID
from ray.util.annotations import PublicAPI
def _stash_least_used_policy(self) -> Policy:
    """Writes the least-recently used policy's state to the Ray object store.

        Also closes the session - if applicable - of the stashed policy.

        Returns:
            The least-recently used policy, that just got removed from the cache.
        """
    dropped_policy_id = self._deque.popleft()
    assert dropped_policy_id in self.cache
    policy = self.cache[dropped_policy_id]
    policy_state = policy.get_state()
    if not self.policy_states_are_swappable:
        self._close_session(policy)
    del self.cache[dropped_policy_id]
    self._policy_state_refs[dropped_policy_id] = ray.put(policy_state)
    return policy