from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateSsdCacheRequest(_messages.Message):
    """The request for UpdateSsdCacheRequest.

  Fields:
    ssdCache: Required. The SSD cache to update, which must always include the
      SSD cache name. Otherwise, only fields mentioned in update_mask need be
      included.
    updateMask: Required. A mask specifying which fields in SsdCache should be
      updated. The update mask must always be specified; this prevents any
      future fields in SsdCache from being erased accidentally by clients that
      do not know about them. Only display_name, size_gib and labels can be
      updated.
  """
    ssdCache = _messages.MessageField('SsdCache', 1)
    updateMask = _messages.StringField(2)