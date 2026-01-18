import time
import logging
import mxnet as mx
from mxnet.module import Module
from .svrg_optimizer import _SVRGOptimizer
def _reset_bind(self):
    """Internal function to reset binded state for both modules."""
    super(SVRGModule, self)._reset_bind()
    self._mod_aux._reset_bind()