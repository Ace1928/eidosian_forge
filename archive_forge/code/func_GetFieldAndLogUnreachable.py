from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
def GetFieldAndLogUnreachable(message, attribute):
    """Response callback to log unreachable while generating fields of the message."""
    if message.unreachable:
        log.warning('The following locations were fully or partially unreachable: {}.'.format(', '.join(message.unreachable)))
    return getattr(message, attribute)