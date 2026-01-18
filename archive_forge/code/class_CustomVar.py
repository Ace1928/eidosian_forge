import os
import unittest.mock
import warnings
import pytest
from packaging import version
import modin.config as cfg
from modin.config.envvars import _check_vars
from modin.config.pubsub import _UNSET, ExactStr
class CustomVar(cfg.EnvironmentVariable, type=request.param):
    """custom var"""
    default = 10
    varname = 'MODIN_CUSTOM'
    choices = (1, 5, 10)