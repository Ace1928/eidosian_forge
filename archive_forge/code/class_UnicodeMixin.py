import sys
from .version import __build__, __version__
class UnicodeMixin(object):
    if sys.version_info >= (3, 0):
        __str__ = lambda x: x.__unicode__()
    else:
        __str__ = lambda x: str(x).encode('utf-8')