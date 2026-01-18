import os
import platform
import pprint
import sys
import time
from io import StringIO
import breezy
from . import bedding, debug, osutils, plugin, trace
def _format_module_list():
    return pprint.pformat(sys.modules)