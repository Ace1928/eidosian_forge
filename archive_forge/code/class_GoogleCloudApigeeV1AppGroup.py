from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1AppGroup(_messages.Message):
    """AppGroup contains the request/response fields representing the logical
  grouping of apps. Note that appgroup_id, create_time and update_time cannot
  be changed by the user, and gets updated by the system. The name and the
  organization once provided cannot be edited subsequently.

  Fields:
    appGroupId: Output only. Internal identifier that cannot be edited
    attributes: A list of attributes
    channelId: channel identifier identifies the owner maintaing this
      grouping.
    channelUri: A reference to the associated storefront/marketplace.
    createdAt: Output only. Created time as milliseconds since epoch.
    displayName: app group name displayed in the UI
    lastModifiedAt: Output only. Modified time as milliseconds since epoch.
    name: Immutable. Name of the AppGroup. Characters you can use in the name
      are restricted to: A-Z0-9._\\-$ %.
    organization: Immutable. the org the app group is created
    status: Valid values are `active` or `inactive`. Note that the status of
      the AppGroup should be updated via UpdateAppGroupRequest by setting the
      action as `active` or `inactive`.
  """
    appGroupId = _messages.StringField(1)
    attributes = _messages.MessageField('GoogleCloudApigeeV1Attribute', 2, repeated=True)
    channelId = _messages.StringField(3)
    channelUri = _messages.StringField(4)
    createdAt = _messages.IntegerField(5)
    displayName = _messages.StringField(6)
    lastModifiedAt = _messages.IntegerField(7)
    name = _messages.StringField(8)
    organization = _messages.StringField(9)
    status = _messages.StringField(10)