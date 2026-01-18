import inspect
import logging
import sys
def _is_method(f):
    return inspect.isfunction(f) or inspect.ismethod(f)