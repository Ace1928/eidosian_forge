import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def _get_quantity(sign, magnitude, unit_suffix):
    res = float('%s%s' % (sign, magnitude))
    if unit_suffix in ['b', 'bit']:
        res /= 8
    return res