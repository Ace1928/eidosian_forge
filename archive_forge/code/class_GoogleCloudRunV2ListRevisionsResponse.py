from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ListRevisionsResponse(_messages.Message):
    """Response message containing a list of Revisions.

  Fields:
    nextPageToken: A token indicating there are more items than page_size. Use
      it in the next ListRevisions request to continue.
    revisions: The resulting list of Revisions.
  """
    nextPageToken = _messages.StringField(1)
    revisions = _messages.MessageField('GoogleCloudRunV2Revision', 2, repeated=True)