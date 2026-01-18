from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsServicesVersionsInstancesDebugRequest(_messages.Message):
    """A AppengineAppsServicesVersionsInstancesDebugRequest object.

  Fields:
    debugInstanceRequest: A DebugInstanceRequest resource to be passed as the
      request body.
    name: Name of the resource requested. Example:
      apps/myapp/services/default/versions/v1/instances/instance-1.
  """
    debugInstanceRequest = _messages.MessageField('DebugInstanceRequest', 1)
    name = _messages.StringField(2, required=True)