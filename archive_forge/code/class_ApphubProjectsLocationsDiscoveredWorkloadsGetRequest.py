from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApphubProjectsLocationsDiscoveredWorkloadsGetRequest(_messages.Message):
    """A ApphubProjectsLocationsDiscoveredWorkloadsGetRequest object.

  Fields:
    name: Required. Fully qualified name of the Discovered Workload to fetch.
      Expected format: `projects/{project}/locations/{location}/discoveredWork
      loads/{discoveredWorkload}`.
  """
    name = _messages.StringField(1, required=True)