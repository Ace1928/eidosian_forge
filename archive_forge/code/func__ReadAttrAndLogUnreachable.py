from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.functions.v2 import util as api_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ReadAttrAndLogUnreachable(message, attribute):
    if message.unreachable:
        log.warning('The following regions were fully or partially unreachable for query: %s\nThis could be due to permission setup. Additional informationcan be found in: https://cloud.google.com/functions/docs/troubleshooting', ', '.join(message.unreachable))
    return getattr(message, attribute)