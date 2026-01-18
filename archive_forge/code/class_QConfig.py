from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
class QConfig(namedtuple('QConfig', ['activation', 'weight'])):
    """
    Describes how to quantize a layer or a part of the network by providing
    settings (observer classes) for activations and weights respectively.


    Note that QConfig needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization preparation function will instantiate observers multiple times for each of the layers.


    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfig(
          activation=MinMaxObserver.with_args(dtype=torch.qint8),
          weight=default_observer.with_args(dtype=torch.qint8))

    """

    def __new__(cls, activation, weight):
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError('QConfig received observer instance, please pass observer class instead. ' + 'Use MyObserver.with_args(x=1) to override arguments to constructor if needed')
        return super().__new__(cls, activation, weight)