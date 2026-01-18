from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsServicesDeleteRequest(_messages.Message):
    """A AppengineAppsServicesDeleteRequest object.

  Fields:
    name: Name of the resource requested. Example:
      apps/myapp/services/default.
  """
    name = _messages.StringField(1, required=True)