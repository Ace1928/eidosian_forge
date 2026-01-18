from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def FormatAppEntry(entry):
    """App Engine formatter for `LogPrinter`.

  Args:
    entry: A log entry message emitted from the V2 API client.

  Returns:
    A string representing the entry or None if there was no text payload.
  """
    if entry.resource.type != 'gae_app':
        return None
    if entry.protoPayload:
        text = six.text_type(entry.protoPayload)
    elif entry.jsonPayload:
        text = six.text_type(entry.jsonPayload)
    else:
        text = entry.textPayload
    service, version = _ExtractServiceAndVersion(entry)
    return '{service}[{version}]  {text}'.format(service=service, version=version, text=text)