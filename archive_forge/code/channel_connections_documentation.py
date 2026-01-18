from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.eventarc import common_publishing
from googlecloudsdk.api_lib.eventarc.base import EventarcClientBase
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
Publish to a Channel Conenction.

    Args:
      channel_connection_ref: Resource, the channel connection to publish from.
      cloud_event: A CloudEvent representation to be passed as the request body.
    