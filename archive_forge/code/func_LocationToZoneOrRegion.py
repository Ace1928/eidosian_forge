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
def LocationToZoneOrRegion(location_id):
    from google.cloud.pubsublite import types
    if len(location_id.split('-')) == 3:
        return types.CloudZone.parse(location_id)
    else:
        return types.CloudRegion.parse(location_id)