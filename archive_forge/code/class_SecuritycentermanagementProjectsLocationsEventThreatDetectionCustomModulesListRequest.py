from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycentermanagementProjectsLocationsEventThreatDetectionCustomModulesListRequest(_messages.Message):
    """A SecuritycentermanagementProjectsLocationsEventThreatDetectionCustomMod
  ulesListRequest object.

  Fields:
    pageSize: Optional. The maximum number of modules to return. The service
      may return fewer than this value. If unspecified, at most 10 configs
      will be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000.
    pageToken: Optional. A page token, received from a previous
      `ListEventThreatDetectionCustomModules` call. Provide this to retrieve
      the subsequent page. When paginating, all other parameters provided to
      `ListEventThreatDetectionCustomModules` must match the call that
      provided the page token.
    parent: Required. Name of parent to list custom modules. Its format is
      "organizations/{organization}/locations/{location}",
      "folders/{folder}/locations/{location}", or
      "projects/{project}/locations/{location}"
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)