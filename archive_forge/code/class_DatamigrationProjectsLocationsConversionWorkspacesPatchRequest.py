from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesPatchRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConversionWorkspacesPatchRequest object.

  Fields:
    conversionWorkspace: A ConversionWorkspace resource to be passed as the
      request body.
    name: Full name of the workspace resource, in the form of: projects/{proje
      ct}/locations/{location}/conversionWorkspaces/{conversion_workspace}.
    requestId: A unique ID used to identify the request. If the server
      receives two requests with the same ID, then the second request is
      ignored. It is recommended to always set this value to a UUID. The ID
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
    updateMask: Required. Field mask is used to specify the fields to be
      overwritten by the update in the conversion workspace resource.
  """
    conversionWorkspace = _messages.MessageField('ConversionWorkspace', 1)
    name = _messages.StringField(2, required=True)
    requestId = _messages.StringField(3)
    updateMask = _messages.StringField(4)