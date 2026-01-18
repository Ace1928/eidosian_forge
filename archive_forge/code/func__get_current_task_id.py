import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
def _get_current_task_id(self) -> TaskID:
    async_task_id = ray._raylet.async_task_id.get()
    if async_task_id is None:
        task_id = self.worker.current_task_id
    else:
        task_id = async_task_id
    return task_id