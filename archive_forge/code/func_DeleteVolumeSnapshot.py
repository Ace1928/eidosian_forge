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
def DeleteVolumeSnapshot(self, resource):
    request = self.messages.BaremetalsolutionProjectsLocationsVolumesSnapshotsDeleteRequest(name=resource.RelativeName())
    return self.snapshots_service.Delete(request)