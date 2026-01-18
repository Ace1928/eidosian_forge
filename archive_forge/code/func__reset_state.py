from __future__ import annotations
import logging
import math
import os
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional
import torch
from .utils import (
from .utils.dataclasses import SageMakerDistributedType
@staticmethod
def _reset_state():
    """Resets `_shared_state`, is used internally and should not be called"""
    GradientState._shared_state.clear()