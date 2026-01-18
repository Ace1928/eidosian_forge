from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def _FallbackFormatter(self, entry):
    if entry.protoPayload:
        return six.text_type(entry.protoPayload)
    elif entry.jsonPayload:
        return six.text_type(entry.jsonPayload)
    else:
        return entry.textPayload