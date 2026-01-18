from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
class DiffAction(object):
    """Enum class representing possible actions to take for an rsync diff."""
    COPY = 'copy'
    REMOVE = 'remove'
    MTIME_SRC_TO_DST = 'mtime_src_to_dst'
    POSIX_SRC_TO_DST = 'posix_src_to_dst'