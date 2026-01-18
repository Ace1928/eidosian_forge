import colorama
from dataclasses import dataclass
import logging
import os
import re
import sys
import threading
import time
from typing import Callable, Dict, List, Set, Tuple, Any, Optional
import ray
from ray.experimental.tqdm_ray import RAY_TQDM_MAGIC
from ray._private.ray_constants import (
from ray.util.debug import log_once
def _warn_once() -> str:
    if log_once('log_dedup_warning'):
        return ' (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/ray-logging.html#log-deduplication for more options.)'
    else:
        return ''