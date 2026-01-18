import abc
import time
import warnings
from collections import namedtuple
from functools import wraps
from typing import Dict, Optional
class MetricsConfig:
    __slots__ = ['params']

    def __init__(self, params: Optional[Dict[str, str]]=None):
        self.params = params
        if self.params is None:
            self.params = {}