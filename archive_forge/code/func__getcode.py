import sys
def _getcode(function):
    try:
        return getattr(function, '__code__')
    except AttributeError:
        return getattr(function, 'func_code', None)