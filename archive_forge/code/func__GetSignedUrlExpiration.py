from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import json
import time
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.diagnose import diagnose_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
def _GetSignedUrlExpiration(self, hours=1):
    """Generate a string expiration time based on some hours in the future.

    Args:
      hours: The number of hours in the future for your timestamp to represent
    Returns:
      A string timestamp measured in seconds since the epoch.
    """
    expiration = datetime.datetime.now() + datetime.timedelta(hours=hours)
    expiration_seconds = time.mktime(expiration.timetuple())
    return six.text_type(int(expiration_seconds))