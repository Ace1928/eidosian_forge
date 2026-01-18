from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.fx import Node
class QuantizationSpecBase(ABC):
    """Base class for different types of quantization specs that allows users to
    specify how to quantize a Tensor (input/output of a Node) in the model
    """
    pass