import inspect
import logging
import sys
def _has_caller(meth):
    return hasattr(meth, 'callers')