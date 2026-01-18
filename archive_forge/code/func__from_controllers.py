import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
@classmethod
def _from_controllers(cls, lib_controllers):
    new_controller = cls.__new__(cls)
    new_controller.lib_controllers = lib_controllers
    return new_controller