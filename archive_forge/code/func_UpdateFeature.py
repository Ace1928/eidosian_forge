from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from typing import Generator
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as messages
def UpdateFeature(self, name, mask, feature):
    """Creates a Feature and returns the long-running operation message.

    Args:
      name: The full resource name in the form
        /projects/*/locations/*/features/*.
      mask: A string list of the field paths to update.
      feature: A Feature message containing the Feature data to update using the
        mask.

    Returns:
      The long running operation reference. Use the feature_waiter and
      OperationRef to watch the operation and get the final status, typically
      using waiter.WaitFor to present a user-friendly spinner.
    """
    req = self.messages.GkehubProjectsLocationsFeaturesPatchRequest(name=name, updateMask=','.join(mask), feature=feature)
    return self.client.projects_locations_features.Patch(req)