from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsListRequest(_messages.Message):
    """A
  ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsListRequest
  object.

  Fields:
    pageSize: Optional. The maximum number of ConfigMaps to return.
    pageToken: Optional. The next_page_token value returned from a previous
      List request, if any.
    parent: Required. List ConfigMaps in the given environment, in the form:
      "projects/{projectId}/locations/{locationId}/environments/{environmentId
      }"
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)