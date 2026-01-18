from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlloydbProjectsLocationsClustersInstancesListRequest(_messages.Message):
    """A AlloydbProjectsLocationsClustersInstancesListRequest object.

  Fields:
    filter: Optional. Filtering results
    orderBy: Optional. Hint for how to order the results
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: A token identifying a page of results the server should return.
    parent: Required. The name of the parent resource. For the required
      format, see the comment on the Instance.name field. Additionally, you
      can perform an aggregated list operation by specifying a value with one
      of the following formats: * projects/{project}/locations/-/clusters/- *
      projects/{project}/locations/{region}/clusters/-
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)