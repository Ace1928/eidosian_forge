from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import os
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.cache import function_result_cache
from googlecloudsdk.core.util import platforms
def _get_default_mode():
    """Gets default permissions files are created with as PosixMode object."""
    max_permissions = 511
    current_umask = os.umask(127)
    os.umask(current_umask)
    mode = max_permissions - current_umask
    mode_without_execution = mode & 438
    return PosixMode.from_base_ten_int(mode_without_execution)