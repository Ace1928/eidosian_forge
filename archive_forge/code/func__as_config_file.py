import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from ray.autoscaler._private import commands
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.event_system import CreateClusterEvent  # noqa: F401
from ray.autoscaler._private.event_system import global_event_system  # noqa: F401
from ray.util.annotations import DeveloperAPI
@contextmanager
@DeveloperAPI
def _as_config_file(cluster_config: Union[dict, str]) -> Iterator[str]:
    if isinstance(cluster_config, dict):
        tmp = tempfile.NamedTemporaryFile('w', prefix='autoscaler-sdk-tmp-')
        tmp.write(json.dumps(cluster_config))
        tmp.flush()
        cluster_config = tmp.name
    if not os.path.exists(cluster_config):
        raise ValueError('Cluster config not found {}'.format(cluster_config))
    yield cluster_config