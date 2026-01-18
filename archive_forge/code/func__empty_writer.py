from __future__ import (absolute_import, division, print_function)
import fcntl
import os
import os.path
import socket as pysocket
from ansible.module_utils.six import PY2
def _empty_writer(msg):
    pass