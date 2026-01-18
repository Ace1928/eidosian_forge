import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
def _is_per_channel_script_obs_instance(module):
    if isinstance(module, torch.jit.RecursiveScriptModule):
        return _is_observer_script_module(module, 'quantization.observer.PerChannelMinMaxObserver') or _is_observer_script_module(module, 'quantization.observer.MovingAveragePerChannelMinMaxObserver')
    return False