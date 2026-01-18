from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListOrgPoliciesRequest(_messages.Message):
    """The request sent to the ListOrgPolicies method.

  Fields:
    pageSize: Size of the pages to be returned. This is currently unsupported
      and will be ignored. The server may at any point start using this field
      to limit page size.
    pageToken: Page token used to retrieve the next page. This is currently
      unsupported and will be ignored. The server may at any point start using
      this field.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)