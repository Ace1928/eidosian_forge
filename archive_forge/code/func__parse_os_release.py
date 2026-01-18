import collections
import os
import re
import sys
import functools
import itertools
def _parse_os_release(lines):
    info = {'NAME': 'Linux', 'ID': 'linux', 'PRETTY_NAME': 'Linux'}
    for line in lines:
        mo = _os_release_line.match(line)
        if mo is not None:
            info[mo.group('name')] = _os_release_unescape.sub('\\1', mo.group('value'))
    return info