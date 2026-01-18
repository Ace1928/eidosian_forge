from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsServicesVersionsDeleteRequest(_messages.Message):
    """A AppengineAppsServicesVersionsDeleteRequest object.

  Fields:
    name: Name of the resource requested. Example:
      apps/myapp/services/default/versions/v1.
  """
    name = _messages.StringField(1, required=True)