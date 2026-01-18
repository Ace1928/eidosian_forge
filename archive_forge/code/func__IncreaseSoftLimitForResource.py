from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import errno
import logging
import multiprocessing
import threading
import traceback
from gslib.utils import constants
from gslib.utils import system_util
from six.moves import queue as Queue
def _IncreaseSoftLimitForResource(resource_name, fallback_value):
    """Sets a new soft limit for the maximum number of open files.

  The soft limit is used for this process (and its children), but the
  hard limit is set by the system and cannot be exceeded.

  We will first try to set the soft limit to the hard limit's value; if that
  fails, we will try to set the soft limit to the fallback_value iff this would
  increase the soft limit.

  Args:
    resource_name: Name of the resource to increase the soft limit for.
    fallback_value: Fallback value to be used if we couldn't set the
                    soft value to the hard value (e.g., if the hard value
                    is "unlimited").

  Returns:
    Current soft limit for the resource (after any changes we were able to
    make), or -1 if the resource doesn't exist.
  """
    try:
        soft_limit, hard_limit = resource.getrlimit(resource_name)
    except (resource.error, ValueError):
        return -1
    if hard_limit > soft_limit:
        try:
            resource.setrlimit(resource_name, (hard_limit, hard_limit))
            return hard_limit
        except (resource.error, ValueError):
            pass
    if soft_limit < fallback_value:
        try:
            resource.setrlimit(resource_name, (fallback_value, hard_limit))
            return fallback_value
        except (resource.error, ValueError):
            return soft_limit
    else:
        return soft_limit