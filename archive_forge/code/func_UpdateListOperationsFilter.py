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
def UpdateListOperationsFilter(resource_ref, args, request):
    """Updates the filter for a ListOperationsRequest."""
    del resource_ref
    if args.filter:
        return request
    if args.subscription:
        request.filter = 'target={}/{}{}'.format(request.name, SUBSCRIPTIONS_RESOURCE_PATH, args.subscription)
    if args.done:
        if request.filter:
            request.filter += ' AND '
        else:
            request.filter = ''
        request.filter += 'done={}'.format(args.done)
    return request