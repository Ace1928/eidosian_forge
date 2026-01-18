from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def GetStr(key):
    return next((x.value.string_value for x in entry.protoPayload.additionalProperties if x.key == key), '-')