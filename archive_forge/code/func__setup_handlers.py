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
def _setup_handlers(create_handler_fn, log):
    debug_handler = _track_handler(create_handler_fn())
    debug_handler.setFormatter(DEFAULT_FORMATTER)
    debug_handler.setLevel(logging.DEBUG)
    log.addHandler(debug_handler)