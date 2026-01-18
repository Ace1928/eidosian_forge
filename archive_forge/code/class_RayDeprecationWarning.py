from typing import Optional
import inspect
import sys
import warnings
from functools import wraps
class RayDeprecationWarning(DeprecationWarning):
    """Specialized Deprecation Warning for fine grained filtering control"""
    pass