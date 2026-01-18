from __future__ import absolute_import, print_function, unicode_literals
import re
import time
import unicodedata
from datetime import datetime
from .enums import ResourceType
from .permissions import Permissions
def _decode_linux_time(mtime):
    return _parse_time(mtime, formats=['%b %d %Y', '%b %d %H:%M'])