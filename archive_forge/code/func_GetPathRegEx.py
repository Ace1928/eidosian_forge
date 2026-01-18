from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def GetPathRegEx(self, subcollection):
    """Returns regex for matching path template."""
    path = self.GetPath(subcollection)
    parts = []
    prev_end = 0
    for match in re.finditer('({[^}]+}/)|({[^}]+})$', path):
        parts.append(path[prev_end:match.start()])
        parts.append('([^/]+)')
        if match.group(1):
            parts.append('/')
        prev_end = match.end()
    if prev_end == len(path):
        parts[-1] = '(.*)$'
    return ''.join(parts)