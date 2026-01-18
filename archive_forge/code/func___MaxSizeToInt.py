import collections
import logging
import re
import textwrap
from apitools.base.py import base_api
from apitools.gen import util
def __MaxSizeToInt(self, max_size):
    """Convert max_size to an int."""
    size_groups = re.match('(?P<size>\\d+)(?P<unit>.B)?$', max_size)
    if size_groups is None:
        raise ValueError('Could not parse maxSize')
    size, unit = size_groups.group('size', 'unit')
    shift = 0
    if unit is not None:
        unit_dict = {'KB': 10, 'MB': 20, 'GB': 30, 'TB': 40}
        shift = unit_dict.get(unit.upper())
        if shift is None:
            raise ValueError('Unknown unit %s' % unit)
    return int(size) * (1 << shift)