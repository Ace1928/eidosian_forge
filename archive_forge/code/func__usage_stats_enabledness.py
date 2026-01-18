import json
import logging
import threading
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.usage.usage_constants as usage_constant
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
from ray.core.generated import usage_pb2, gcs_pb2
def _usage_stats_enabledness() -> UsageStatsEnabledness:
    usage_stats_enabled_env_var = os.getenv(usage_constant.USAGE_STATS_ENABLED_ENV_VAR)
    if usage_stats_enabled_env_var == '0':
        return UsageStatsEnabledness.DISABLED_EXPLICITLY
    elif usage_stats_enabled_env_var == '1':
        return UsageStatsEnabledness.ENABLED_EXPLICITLY
    elif usage_stats_enabled_env_var is not None:
        raise ValueError(f'Valid value for {usage_constant.USAGE_STATS_ENABLED_ENV_VAR} env var is 0 or 1, but got {usage_stats_enabled_env_var}')
    usage_stats_enabled_config_var = None
    try:
        with open(_usage_stats_config_path()) as f:
            config = json.load(f)
            usage_stats_enabled_config_var = config.get('usage_stats')
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.debug(f'Failed to load usage stats config {e}')
    if usage_stats_enabled_config_var is False:
        return UsageStatsEnabledness.DISABLED_EXPLICITLY
    elif usage_stats_enabled_config_var is True:
        return UsageStatsEnabledness.ENABLED_EXPLICITLY
    elif usage_stats_enabled_config_var is not None:
        raise ValueError(f"Valid value for 'usage_stats' in {_usage_stats_config_path()} is true or false, but got {usage_stats_enabled_config_var}")
    return UsageStatsEnabledness.ENABLED_BY_DEFAULT