from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIdentityAccesscontextmanagerV1MethodSelector(_messages.Message):
    """An allowed method or permission of a service specified in ApiOperation.

  Fields:
    method: A valid method name for the corresponding `service_name` in
      ApiOperation. If `*` is used as the value for the `method`, then ALL
      methods and permissions are allowed.
    permission: A valid Cloud IAM permission for the corresponding
      `service_name` in ApiOperation.
  """
    method = _messages.StringField(1)
    permission = _messages.StringField(2)