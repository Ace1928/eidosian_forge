from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from concurrent import futures
import time
from typing import Optional
from google.cloud.pubsublite import cloudpubsub
from google.cloud.pubsublite import types
from google.pubsub_v1 import PubsubMessage
from googlecloudsdk.command_lib.pubsub import lite_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import gapic_util
from googlecloudsdk.core import log
from six.moves import queue
def _SubscriptionResourceToPath(self, resource):
    return types.SubscriptionPath(project=lite_util.ProjectIdToProjectNumber(resource.projectsId), location=lite_util.LocationToZoneOrRegion(resource.locationsId), name=resource.subscriptionsId)