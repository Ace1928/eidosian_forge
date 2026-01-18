from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesUndeleteRequest(_messages.Message):
    """A ServicemanagementServicesUndeleteRequest object.

  Fields:
    serviceName: Required. The name of the service. See the
      [overview](https://cloud.google.com/service-management/overview) for
      naming requirements. For example: `example.googleapis.com`.
  """
    serviceName = _messages.StringField(1, required=True)