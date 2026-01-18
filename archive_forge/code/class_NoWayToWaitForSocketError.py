import errno
import select
import sys
from functools import partial
class NoWayToWaitForSocketError(Exception):
    pass