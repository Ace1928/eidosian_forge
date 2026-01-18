import time
import logging
import datetime
import functools
from pyzor.engines.common import *
@staticmethod
def _real_key(key):
    return '%s.%s' % (NAMESPACE, key)