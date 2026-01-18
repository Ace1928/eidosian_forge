import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
def _is_activation_post_process(module):
    return isinstance(module, (torch.ao.quantization.ObserverBase, torch.ao.quantization.FakeQuantizeBase)) or _is_observer_script_module(module, 'quantization.observer')