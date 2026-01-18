from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import hashlib
import os
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.storage import storage_parallel
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import times
from googlecloudsdk.third_party.appengine.tools import context_util
from six.moves import map  # pylint: disable=redefined-builtin
def _IsTTLSafe(ttl, obj):
    """Determines whether a GCS object is close to end-of-life.

  In order to reduce false negative rate (objects that are close to deletion but
  aren't marked as such) the returned filter is forward-adjusted with
  _TTL_MARGIN.

  Args:
    ttl: datetime.timedelta, TTL of objects, or None if no TTL.
    obj: storage object to check.

  Returns:
    True if the ojbect is safe or False if it is approaching end of life.
  """
    if ttl is None:
        return True
    now = times.Now(times.UTC)
    delta = ttl - _TTL_MARGIN
    return now - obj.timeCreated <= delta