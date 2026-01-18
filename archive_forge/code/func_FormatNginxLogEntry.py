from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def FormatNginxLogEntry(entry):
    """App Engine nginx.* formatter for `LogPrinter`.

  Args:
    entry: A log entry message emitted from the V2 API client.

  Returns:
    A string representing the entry if it is a request entry.
  """
    if entry.resource.type != 'gae_app':
        return None
    log_id = util.ExtractLogId(entry.logName)
    if log_id not in NGINX_LOGS:
        return None
    service, version = _ExtractServiceAndVersion(entry)
    msg = '"{method} {resource}" {status}'.format(method=entry.httpRequest.requestMethod or '-', resource=entry.httpRequest.requestUrl or '-', status=entry.httpRequest.status or '-')
    return '{service}[{version}]  {msg}'.format(service=service, version=version, msg=msg)