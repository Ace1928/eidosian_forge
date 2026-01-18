import os
import sys
from os_ken.lib import hub
from os_ken import cfg
import logging
from os_ken import log
from os_ken import flags
from os_ken import __version__ as version
from os_ken.base.app_manager import AppManager
from os_ken.controller import controller
from os_ken.topology import switches
def _parse_user_flags():
    """
    Parses user-flags file and loads it to register user defined options.
    """
    try:
        idx = list(sys.argv).index('--user-flags')
        user_flags_file = sys.argv[idx + 1]
    except (ValueError, IndexError):
        user_flags_file = ''
    if user_flags_file and os.path.isfile(user_flags_file):
        from os_ken.utils import _import_module_file
        _import_module_file(user_flags_file)