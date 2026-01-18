from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsServicesVersionsCreateRequest(_messages.Message):
    """A AppengineAppsServicesVersionsCreateRequest object.

  Fields:
    parent: Name of the parent resource to create this version under. Example:
      apps/myapp/services/default.
    version: A Version resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    version = _messages.MessageField('Version', 2)