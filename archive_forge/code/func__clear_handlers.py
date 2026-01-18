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
def _clear_handlers(log):
    to_remove = [handler for handler in log.handlers if _is_torch_handler(handler)]
    for handler in to_remove:
        log.removeHandler(handler)