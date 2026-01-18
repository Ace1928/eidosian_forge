from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import threading
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.debug import errors
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
import six
from six.moves import urllib
def LogQueryV2String(breakpoint, separator=' '):
    """Returns an advanced log query string for use with gcloud logging read.

  Args:
    breakpoint: A breakpoint object with added information on project, service,
      and debug target.
    separator: A string to append between conditions
  Returns:
    A log query suitable for use with gcloud logging read.
  Raises:
    InvalidLogFormatException if the breakpoint has an invalid log expression.
  """
    query = 'resource.type=gae_app{sep}logName:request_log{sep}resource.labels.module_id="{service}"{sep}resource.labels.version_id="{version}"{sep}severity={logLevel}'.format(service=breakpoint.service, version=breakpoint.version, logLevel=breakpoint.logLevel or 'INFO', sep=separator)
    if breakpoint.logMessageFormat:
        query += '{sep}"{text}"'.format(text=re.sub('\\$([0-9]+)', '" "', SplitLogExpressions(breakpoint.logMessageFormat)[0]), sep=separator)
    return query