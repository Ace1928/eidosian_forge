import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid
from traceback import format_exc
from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
def MakeProxyType(name, exposed, _cache={}):
    """
    Return a proxy type whose methods are given by `exposed`
    """
    exposed = tuple(exposed)
    try:
        return _cache[name, exposed]
    except KeyError:
        pass
    dic = {}
    for meth in exposed:
        exec('def %s(self, /, *args, **kwds):\n        return self._callmethod(%r, args, kwds)' % (meth, meth), dic)
    ProxyType = type(name, (BaseProxy,), dic)
    ProxyType._exposed_ = exposed
    _cache[name, exposed] = ProxyType
    return ProxyType