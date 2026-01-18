import contextlib
import copy
import pickle
import unittest
from types import FunctionType, ModuleType
from typing import Any, Dict, Set
from unittest import mock
class SubConfigProxy:
    """
    Shim to redirect to main config.
    `config.triton.cudagraphs` maps to _config["triton.cudagraphs"]
    """

    def __init__(self, config, prefix):
        super().__setattr__('_config', config)
        super().__setattr__('_prefix', prefix)

    def __setattr__(self, name, value):
        return self._config.__setattr__(self._prefix + name, value)

    def __getattr__(self, name):
        return self._config.__getattr__(self._prefix + name)

    def __delattr__(self, name):
        return self._config.__delattr__(self._prefix + name)