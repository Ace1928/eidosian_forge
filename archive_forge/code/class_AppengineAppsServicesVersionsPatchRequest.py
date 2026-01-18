from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsServicesVersionsPatchRequest(_messages.Message):
    """A AppengineAppsServicesVersionsPatchRequest object.

  Fields:
    name: Name of the resource to update. Example:
      apps/myapp/services/default/versions/1.
    updateMask: Standard field mask for the set of fields to be updated.
    version: A Version resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    version = _messages.MessageField('Version', 3)