import uuid
from collections import OrderedDict
from functools import wraps
from typing import Callable, Dict, List, Optional, Type
import torch.nn as nn
from torch.distributed._composable_state import _State
class RegistryItem:
    pass