import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
def _get_actor_call_stats(self):
    """Get the current worker's task counters.

        Returns:
            A dictionary keyed by the function name. The values are
            dictionaries with form ``{"pending": 0, "running": 1,
            "finished": 2}``.
        """
    worker = self.worker
    worker.check_connected()
    return worker.core_worker.get_actor_call_stats()