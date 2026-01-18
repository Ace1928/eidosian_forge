from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ModifyEntryOverviewRequest(_messages.Message):
    """Request message for ModifyEntryOverview.

  Fields:
    entryOverview: Required. The new value for the Entry Overview.
  """
    entryOverview = _messages.MessageField('GoogleCloudDatacatalogV1EntryOverview', 1)