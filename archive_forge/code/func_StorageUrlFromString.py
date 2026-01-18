from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import stat
import sys
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.utils import system_util
from gslib.utils import text_util
def StorageUrlFromString(url_str):
    """Static factory function for creating a StorageUrl from a string."""
    scheme = GetSchemeFromUrlString(url_str)
    if not IsKnownUrlScheme(scheme):
        raise InvalidUrlError('Unrecognized scheme "%s"' % scheme)
    if scheme == 'file':
        path = _GetPathFromUrlString(url_str)
        is_stream = path == '-'
        is_fifo = False
        try:
            is_fifo = stat.S_ISFIFO(os.stat(path).st_mode)
        except OSError:
            pass
        return _FileUrl(url_str, is_stream=is_stream, is_fifo=is_fifo)
    return _CloudUrl(url_str)