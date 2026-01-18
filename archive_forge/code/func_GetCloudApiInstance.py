from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import re
import six
from gslib.cloud_api import ArgumentException
from gslib.utils.text_util import AddQueryParamToUrl
def GetCloudApiInstance(cls, thread_state=None):
    """Gets a gsutil Cloud API instance.

  Since Cloud API implementations are not guaranteed to be thread-safe, each
  thread needs its own instance. These instances are passed to each thread
  via the thread pool logic in command.

  Args:
    cls: Command class to be used for single-threaded case.
    thread_state: Per thread state from this thread containing a gsutil
                  Cloud API instance.

  Returns:
    gsutil Cloud API instance.
  """
    return thread_state or cls.gsutil_api