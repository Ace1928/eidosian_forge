from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgecontainerProjectsLocationsMachinesListRequest(_messages.Message):
    """A EdgecontainerProjectsLocationsMachinesListRequest object.

  Fields:
    filter: Only resources matching this filter will be listed.
    orderBy: Specifies the order in which resources will be listed.
    pageSize: The maximum number of resources to list.
    pageToken: A page token received from previous list request.
    parent: Required. The parent site, which owns this collection of machines.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)