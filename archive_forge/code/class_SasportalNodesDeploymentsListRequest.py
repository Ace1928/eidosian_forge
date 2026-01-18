from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalNodesDeploymentsListRequest(_messages.Message):
    """A SasportalNodesDeploymentsListRequest object.

  Fields:
    filter: The filter expression. The filter should have the following
      format: "DIRECT_CHILDREN" or format: "direct_children". The filter is
      case insensitive. If empty, then no deployments are filtered.
    pageSize: The maximum number of deployments to return in the response.
    pageToken: A pagination token returned from a previous call to
      ListDeployments that indicates where this listing should continue from.
    parent: Required. The parent resource name, for example, "nodes/1",
      customer/1/nodes/2.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)