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
def ListFeatures(self, parent):
    """Lists Features from the Hub API.

    Args:
      parent: The parent in the form /projects/*/locations/*.

    Returns:
      A list of Features.
    """
    req = self.messages.GkehubProjectsLocationsFeaturesListRequest(parent=parent)
    resp = self.client.projects_locations_features.List(req)
    return resp.resources