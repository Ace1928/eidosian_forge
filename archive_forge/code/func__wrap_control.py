import errno
import math
import select
import socket
import sys
import time
from collections import namedtuple
from ansible.module_utils.six.moves.collections_abc import Mapping
def _wrap_control(self, changelist, max_events, timeout):
    return self._kqueue.control(changelist, max_events, timeout)