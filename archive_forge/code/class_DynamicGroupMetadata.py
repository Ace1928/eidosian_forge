from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DynamicGroupMetadata(_messages.Message):
    """Dynamic group metadata like queries and status.

  Fields:
    queries: Memberships will be the union of all queries. Only one entry with
      USER resource is currently supported. Customers can create up to 500
      dynamic groups.
    status: Output only. Status of the dynamic group.
  """
    queries = _messages.MessageField('DynamicGroupQuery', 1, repeated=True)
    status = _messages.MessageField('DynamicGroupStatus', 2)