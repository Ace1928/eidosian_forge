import gc
from io import StringIO
import weakref
from pyomo.common.unittest import TestCase
from pyomo.common.log import LoggingIntercept
from pyomo.common.plugin_base import (
class myFoo(Plugin):
    implements(ICustomGone)