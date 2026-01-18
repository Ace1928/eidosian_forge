from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateFeedRequest(_messages.Message):
    """Create asset feed request.

  Fields:
    feed: Required. The feed details. The field `name` must be empty and it
      will be generated in the format of:
      projects/project_number/feeds/feed_id
      folders/folder_number/feeds/feed_id
      organizations/organization_number/feeds/feed_id
    feedId: Required. This is the client-assigned asset feed identifier and it
      needs to be unique under a specific parent project/folder/organization.
  """
    feed = _messages.MessageField('Feed', 1)
    feedId = _messages.StringField(2)