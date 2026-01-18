from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApikeysProjectsKeysPatchRequest(_messages.Message):
    """A ApikeysProjectsKeysPatchRequest object.

  Fields:
    name: Required. The resource name of the API key to be modified.
    updateMask: Required. The field mask specifies which fields should be
      updated as part of this request. All other fields will be ignored.
      Allowed field mask: `display_name` and `restrictions`
    v2alpha1ApiKey: A V2alpha1ApiKey resource to be passed as the request
      body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    v2alpha1ApiKey = _messages.MessageField('V2alpha1ApiKey', 3)