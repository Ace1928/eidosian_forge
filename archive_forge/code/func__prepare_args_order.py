from __future__ import absolute_import, division, print_function
import os
from functools import wraps
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six import iteritems
@staticmethod
def _prepare_args_order(order):
    return tuple(order) if is_sequence(order) else tuple(order.split())