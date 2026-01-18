from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesPatchConfigRequest(_messages.Message):
    """A ServicemanagementServicesPatchConfigRequest object.

  Fields:
    service: A Service resource to be passed as the request body.
    serviceName: The name of the service.  See the `ServiceManager` overview
      for naming requirements.  For example: `example.googleapis.com`.
    updateMask: A mask specifying which fields to update.
  """
    service = _messages.MessageField('Service', 1)
    serviceName = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)