from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsTargetsGetRequest(_messages.Message):
    """A ClouddeployProjectsLocationsTargetsGetRequest object.

  Fields:
    name: Required. Name of the `Target`. Format must be
      `projects/{project_id}/locations/{location_name}/targets/{target_name}`.
  """
    name = _messages.StringField(1, required=True)