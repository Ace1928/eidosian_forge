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
def _is_valid_module(qname):
    try:
        __import__(qname)
        return True
    except ImportError:
        return False