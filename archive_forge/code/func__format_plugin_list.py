import os
import platform
import pprint
import sys
import time
from io import StringIO
import breezy
from . import bedding, debug, osutils, plugin, trace
def _format_plugin_list():
    return ''.join(plugin.describe_plugins(show_paths=True))