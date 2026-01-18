from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as apilib_exceptions
from googlecloudsdk.command_lib.bms import util
import six
def ListSnapshotSchedulePolicies(self, project_resource, limit=None, page_size=None):
    parent = 'projects/%s/locations/global' % project_resource
    request = self.messages.BaremetalsolutionProjectsLocationsSnapshotSchedulePoliciesListRequest(parent=parent)
    return list_pager.YieldFromList(self.snapshot_schedule_policies_service, request, limit=limit, batch_size_attribute='pageSize', batch_size=page_size, field='snapshotSchedulePolicies')