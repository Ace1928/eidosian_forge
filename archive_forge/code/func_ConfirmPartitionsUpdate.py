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
def ConfirmPartitionsUpdate(resource_ref, args, request):
    """Prompts to confirm an update to a topic's partition count."""
    del resource_ref
    if 'partitions' not in args or not args.partitions:
        return request
    console_io.PromptContinue(message='Warning: The number of partitions in a topic can be increased but not decreased. Additionally message order is not guaranteed across a topic resize. See https://cloud.google.com/pubsub/lite/docs/topics#scaling_capacity for more details', default=True, cancel_on_no=True)
    return request