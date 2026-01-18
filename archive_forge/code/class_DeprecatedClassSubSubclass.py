import sys
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
from pyomo.common.log import LoggingIntercept
from io import StringIO
import logging
class DeprecatedClassSubSubclass(DeprecatedClassSubclass):
    attr = 'DeprecatedClassSubSubclass'