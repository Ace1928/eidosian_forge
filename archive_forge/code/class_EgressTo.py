from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EgressTo(_messages.Message):
    """Defines the conditions under which an EgressPolicy matches a request.
  Conditions are based on information about the ApiOperation intended to be
  performed on the `resources` specified. Note that if the destination of the
  request is also protected by a ServicePerimeter, then that ServicePerimeter
  must have an IngressPolicy which allows access in order for this request to
  succeed. The request must match `operations` AND `resources` fields in order
  to be allowed egress out of the perimeter.

  Fields:
    externalResources: A list of external resources that are allowed to be
      accessed. A request matches if it contains an external resource in this
      list (Example: s3://bucket/path). Currently '*' is not allowed.
    operations: A list of ApiOperations allowed to be performed by the sources
      specified in the corresponding EgressFrom. A request matches if it uses
      an operation/service in this list.
    resources: A list of resources, currently only projects in the form
      `projects/`, that are allowed to be accessed by sources defined in the
      corresponding EgressFrom. A request matches if it contains a resource in
      this list. If `*` is specified for `resources`, then this EgressTo rule
      will authorize access to all resources outside the perimeter.
    roles: IAM roles that represent the set of operations allowed to be
      performed by the sources specified in the corresponding EgressFrom.
  """
    externalResources = _messages.StringField(1, repeated=True)
    operations = _messages.MessageField('ApiOperation', 2, repeated=True)
    resources = _messages.StringField(3, repeated=True)
    roles = _messages.StringField(4, repeated=True)