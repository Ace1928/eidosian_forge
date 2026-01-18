import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def _load_plugin_module(name, dir):
    """Load plugin by name.

    Args:
      name: The plugin name in the breezy.plugins namespace.
      dir: The directory the plugin is loaded from for error messages.
    """
    if _MODULE_PREFIX + name in sys.modules:
        return
    try:
        __import__(_MODULE_PREFIX + name)
    except errors.IncompatibleVersion as e:
        warning_message = 'Unable to load plugin %r. It supports %s versions %r but the current version is %s' % (name, e.api.__name__, e.wanted, e.current)
        return record_plugin_warning(warning_message)
    except Exception as e:
        trace.log_exception_quietly()
        if 'error' in debug.debug_flags:
            trace.print_exception(sys.exc_info(), sys.stderr)
        return record_plugin_warning('Unable to load plugin {!r} from {!r}: {}'.format(name, dir, e))