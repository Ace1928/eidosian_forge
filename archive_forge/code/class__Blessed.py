from __future__ import (absolute_import, division, print_function)
from functools import reduce
import inspect
import math
import operator
import sys
from pkg_resources import parse_requirements, parse_version
import numpy as np
import pytest
class _Blessed(object):
    pass