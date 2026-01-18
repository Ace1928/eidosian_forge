from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsCustomTargetTypesGetRequest(_messages.Message):
    """A ClouddeployProjectsLocationsCustomTargetTypesGetRequest object.

  Fields:
    name: Required. Name of the `CustomTargetType`. Format must be `projects/{
      project_id}/locations/{location_name}/customTargetTypes/{custom_target_t
      ype}`.
  """
    name = _messages.StringField(1, required=True)