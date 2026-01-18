import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def _get_constant(unit_prefix, unit_system):
    if not unit_prefix:
        return 1
    elif unit_system == 'SI':
        res = getattr(units, unit_prefix)
    elif unit_system == 'IEC':
        if unit_prefix.endswith('i'):
            res = getattr(units, unit_prefix)
        else:
            res = getattr(units, '%si' % unit_prefix)
    elif unit_system == 'mixed':
        if unit_prefix == 'K':
            unit_prefix = 'k'
        res = getattr(units, unit_prefix)
    return res