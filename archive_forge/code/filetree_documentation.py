from __future__ import (absolute_import, division, print_function)
import os
import pwd
import grp
import stat
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.utils.display import Display
 Returns dictionary with file properties, or return None on failure 