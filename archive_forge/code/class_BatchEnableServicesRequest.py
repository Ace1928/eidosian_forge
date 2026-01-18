from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchEnableServicesRequest(_messages.Message):
    """Request message for the `BatchEnableServices` method.

  Fields:
    serviceIds: The identifiers of the services to enable on the project. A
      valid identifier would be: serviceusage.googleapis.com Enabling services
      requires that each service is public or is shared with the user enabling
      the service. A single request can enable a maximum of 20 services at a
      time. If more than 20 services are specified, the request will fail, and
      no state changes will occur.
  """
    serviceIds = _messages.StringField(1, repeated=True)