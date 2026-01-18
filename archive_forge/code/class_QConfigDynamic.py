from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
class QConfigDynamic(namedtuple('QConfigDynamic', ['activation', 'weight'])):
    """
    Describes how to dynamically quantize a layer or a part of the network by providing
    settings (observer classes) for weights.

    It's like QConfig, but for dynamic quantization.

    Note that QConfigDynamic needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of the layers.

    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfigDynamic(weight=default_observer.with_args(dtype=torch.qint8))
    """

    def __new__(cls, activation=torch.nn.Identity, weight=torch.nn.Identity):
        if isinstance(weight, nn.Module):
            raise ValueError('QConfigDynamic received observer instance, please pass observer class instead. ' + 'Use MyObserver.with_args(x=1) to override arguments to constructor if needed')
        warnings.warn('QConfigDynamic is going to be deprecated in PyTorch 1.12, please use QConfig instead')
        return super().__new__(cls, activation, weight)