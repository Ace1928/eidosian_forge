import importlib
from functools import wraps
from typing import Protocol, runtime_checkable
import numpy as np
from scipy.sparse import issparse
from .._config import get_config
from ._available_if import available_if
class ContainerAdaptersManager:

    def __init__(self):
        self.adapters = {}

    @property
    def supported_outputs(self):
        return {'default'} | set(self.adapters)

    def register(self, adapter):
        self.adapters[adapter.container_lib] = adapter