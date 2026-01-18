from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def SetSeekTarget(resource_ref, args, request):
    """Sets the target for a SeekSubscriptionRequest."""
    del resource_ref
    psl = PubsubLiteMessages()
    request.seekSubscriptionRequest = GetSeekRequest(args, psl)
    log.warning('The seek operation will complete once subscribers react to the seek. ' + 'If subscribers are offline, `pubsub lite-operations describe` can be ' + 'used to check the operation status later.')
    return request