from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
def _get_message_module(version):
    """Returns the message module for the Compute API."""
    return apis.GetMessagesModule('compute', apis.ResolveVersion('compute', version))