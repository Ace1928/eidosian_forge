import sys
def _getfuncdict(function):
    return getattr(function, '__dict__', None)