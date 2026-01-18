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
def ListRollouts(self, req: messages.GkehubProjectsLocationsRolloutsListRequest, page_size=None, limit=None) -> Generator[messages.Rollout, None, None]:
    """Lists fleet rollouts."""
    return list_pager.YieldFromList(self.client.projects_locations_rollouts, req, field='rollouts', batch_size=page_size, limit=limit, batch_size_attribute='pageSize')