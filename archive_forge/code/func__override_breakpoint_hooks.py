import logging
import os
import sys
import threading
import importlib
import ray
from ray.util.annotations import DeveloperAPI
def _override_breakpoint_hooks():
    """
    This method overrides the breakpoint() function to set_trace()
    so that other threads can reuse the same setup logic.
    This is based on: https://github.com/microsoft/debugpy/blob/ef9a67fe150179ee4df9997f9273723c26687fab/src/debugpy/_vendored/pydevd/pydev_sitecustomize/sitecustomize.py#L87 # noqa: E501
    """
    sys.__breakpointhook__ = set_trace
    sys.breakpointhook = set_trace
    import builtins as __builtin__
    __builtin__.breakpoint = set_trace