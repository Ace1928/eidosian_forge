import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
class MappingTestError(Exception):
    pass