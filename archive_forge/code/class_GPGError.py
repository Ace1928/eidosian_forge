from __future__ import (absolute_import, division, print_function)
import abc
import os
from ansible.module_utils import six
class GPGError(Exception):
    pass