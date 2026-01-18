import logging
import re
from collections import defaultdict
from typing import List, Optional, Dict, AsyncIterable, Tuple, Callable
from ray.dashboard.modules.job.common import JOB_LOGS_PATH_TEMPLATE
from ray.util.state.common import (
from ray.util.state.exception import DataSourceUnavailable
from ray.util.state.state_manager import StateDataSourceClient
from ray._private.pydantic_compat import BaseModel
from ray.dashboard.datacenter import DataSource
def _categorize_log_files(self, log_files: List[str]) -> Dict[str, List[str]]:
    """Categorize the given log files after filterieng them out using a given glob.

        Returns:
            Dictionary of {component_name -> list of log files}
        """
    result = defaultdict(list)
    for log_file in log_files:
        if 'worker' in log_file and log_file.endswith('.out'):
            result['worker_out'].append(log_file)
        elif 'worker' in log_file and log_file.endswith('.err'):
            result['worker_err'].append(log_file)
        elif 'core-worker' in log_file and log_file.endswith('.log'):
            result['core_worker'].append(log_file)
        elif 'core-driver' in log_file and log_file.endswith('.log'):
            result['driver'].append(log_file)
        elif 'raylet.' in log_file:
            result['raylet'].append(log_file)
        elif 'gcs_server.' in log_file:
            result['gcs_server'].append(log_file)
        elif 'log_monitor' in log_file:
            result['internal'].append(log_file)
        elif 'monitor' in log_file:
            result['autoscaler'].append(log_file)
        elif 'agent.' in log_file:
            result['agent'].append(log_file)
        elif 'dashboard.' in log_file:
            result['dashboard'].append(log_file)
        else:
            result['internal'].append(log_file)
    return result