from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2StoredType(_messages.Message):
    """A reference to a StoredInfoType to use with scanning.

  Fields:
    createTime: Timestamp indicating when the version of the `StoredInfoType`
      used for inspection was created. Output-only field, populated by the
      system.
    name: Resource name of the requested `StoredInfoType`, for example
      `organizations/433245324/storedInfoTypes/432452342` or
      `projects/project-id/storedInfoTypes/432452342`.
  """
    createTime = _messages.StringField(1)
    name = _messages.StringField(2)