from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesPatchRequest(_messages.Message):
    """A ServicemanagementServicesPatchRequest object.

  Fields:
    managedService: A ManagedService resource to be passed as the request
      body.
    serviceName: The name of the service.  See the `ServiceManager` overview
      for naming requirements.  For example: `example.googleapis.com`.
    updateMask: A mask specifying which fields to update.
  """
    managedService = _messages.MessageField('ManagedService', 1)
    serviceName = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)