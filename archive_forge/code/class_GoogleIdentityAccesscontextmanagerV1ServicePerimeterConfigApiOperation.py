from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIdentityAccesscontextmanagerV1ServicePerimeterConfigApiOperation(_messages.Message):
    """Identification for an API Operation.

  Fields:
    methodSelectors: API methods or permissions to allow. Method or permission
      must belong to the service specified by `service_name` field. A single
      MethodSelector entry with `*` specified for the `method` field will
      allow all methods AND permissions for the service specified in
      `service_name`.
    serviceName: The name of the API whose methods or permissions the
      IngressPolicy or EgressPolicy want to allow. A single ApiOperation with
      `service_name` field set to `*` will allow all methods AND permissions
      for all services.
  """
    methodSelectors = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1ServicePerimeterConfigMethodSelector', 1, repeated=True)
    serviceName = _messages.StringField(2)