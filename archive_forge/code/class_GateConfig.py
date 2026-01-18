import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union
import torch
from xformers.components import Activation
from xformers.components.feedforward import (
class GateConfig(str, Enum):
    RoundRobin = 'round_robin'
    Top2 = 'top_2'