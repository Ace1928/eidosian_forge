from __future__ import absolute_import, print_function, unicode_literals
import re
import time
import unicodedata
from datetime import datetime
from .enums import ResourceType
from .permissions import Permissions
Decode a Windows NT FTP LIST line.

    Examples:
        Decode a directory line::

            >>> line = "11-02-18  02:12PM       <DIR>          images"
            >>> match = RE_WINDOWSNT.match(line)
            >>> pprint(decode_windowsnt(line, match))
            {'basic': {'is_dir': True, 'name': 'images'},
             'details': {'modified': 1518358320.0, 'type': 1},
             'ftp': {'ls': '11-02-18  02:12PM       <DIR>          images'}}

        Decode a file line::

            >>> line = "11-02-18  03:33PM                 9276 logo.gif"
            >>> match = RE_WINDOWSNT.match(line)
            >>> pprint(decode_windowsnt(line, match))
            {'basic': {'is_dir': False, 'name': 'logo.gif'},
             'details': {'modified': 1518363180.0, 'size': 9276, 'type': 2},
             'ftp': {'ls': '11-02-18  03:33PM                 9276 logo.gif'}}

        Alternatively, the time might also be present in 24-hour format::

            >>> line = "11-02-18  15:33                   9276 logo.gif"
            >>> match = RE_WINDOWSNT.match(line)
            >>> decode_windowsnt(line, match)["details"]["modified"]
            1518363180.0

    