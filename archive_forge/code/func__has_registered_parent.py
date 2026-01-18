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
def _has_registered_parent(log_qname):
    cur_log = logging.getLogger(log_qname)
    registered_log_qnames = log_registry.get_log_qnames()
    while cur_log.parent:
        if cur_log.name in registered_log_qnames:
            return True
        cur_log = cur_log.parent
    return False