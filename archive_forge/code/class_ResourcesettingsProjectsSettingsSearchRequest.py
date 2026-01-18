from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResourcesettingsProjectsSettingsSearchRequest(_messages.Message):
    """A ResourcesettingsProjectsSettingsSearchRequest object.

  Fields:
    pageSize: Unused. The size of the page to be returned.
    pageToken: Unused. A page token used to retrieve the next page.
    parent: The Cloud resource that parents the setting. Must be in one of the
      following forms: * `projects/{project_number}` * `projects/{project_id}`
      * `folders/{folder_id}` * `organizations/{organization_id}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)