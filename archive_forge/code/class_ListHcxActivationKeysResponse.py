from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListHcxActivationKeysResponse(_messages.Message):
    """Response message for VmwareEngine.ListHcxActivationKeys

  Fields:
    hcxActivationKeys: List of HCX activation keys.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    unreachable: Locations that could not be reached when making an aggregated
      query using wildcards.
  """
    hcxActivationKeys = _messages.MessageField('HcxActivationKey', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)