from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def FormatRequestLogEntry(entry):
    """App Engine request_log formatter for `LogPrinter`.

  Args:
    entry: A log entry message emitted from the V2 API client.

  Returns:
    A string representing the entry if it is a request entry.
  """
    if entry.resource.type != 'gae_app':
        return None
    log_id = util.ExtractLogId(entry.logName)
    if log_id != 'appengine.googleapis.com/request_log':
        return None
    service, version = _ExtractServiceAndVersion(entry)

    def GetStr(key):
        return next((x.value.string_value for x in entry.protoPayload.additionalProperties if x.key == key), '-')

    def GetInt(key):
        return next((x.value.integer_value for x in entry.protoPayload.additionalProperties if x.key == key), '-')
    msg = '"{method} {resource} {http_version}" {status}'.format(method=GetStr('method'), resource=GetStr('resource'), http_version=GetStr('httpVersion'), status=GetInt('status'))
    return '{service}[{version}]  {msg}'.format(service=service, version=version, msg=msg)