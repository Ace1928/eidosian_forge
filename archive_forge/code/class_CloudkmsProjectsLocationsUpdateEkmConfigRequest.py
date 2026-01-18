from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsUpdateEkmConfigRequest(_messages.Message):
    """A CloudkmsProjectsLocationsUpdateEkmConfigRequest object.

  Fields:
    ekmConfig: A EkmConfig resource to be passed as the request body.
    name: Output only. The resource name for the EkmConfig in the format
      `projects/*/locations/*/ekmConfig`.
    updateMask: Required. List of fields to be updated in this request.
  """
    ekmConfig = _messages.MessageField('EkmConfig', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)