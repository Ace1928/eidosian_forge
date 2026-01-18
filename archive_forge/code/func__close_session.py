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
@staticmethod
def _close_session(policy: Policy):
    sess = policy.get_session()
    if sess is not None:
        sess.close()