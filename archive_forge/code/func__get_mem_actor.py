from io import StringIO
from typing import Dict, List
import ray
from ray.data.context import DataContext
def _get_mem_actor():
    return _MemActor.options(name='mem_tracing_actor', get_if_exists=True, lifetime='detached').remote()