from __future__ import absolute_import, print_function, unicode_literals
import re
import time
import unicodedata
from datetime import datetime
from .enums import ResourceType
from .permissions import Permissions
def _decode_windowsnt_time(mtime):
    return _parse_time(mtime, formats=['%d-%m-%y %I:%M%p', '%d-%m-%y %H:%M'])