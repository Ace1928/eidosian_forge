import functools
import itertools
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import __import__
from typing import Dict, List, Optional, Set, Union
from weakref import WeakSet
import torch._guards
import torch.distributed as dist
def _update_log_state_from_env():
    global log_state
    log_setting = os.environ.get(LOG_ENV_VAR, None)
    if log_setting is not None:
        log_state = _parse_log_settings(log_setting)