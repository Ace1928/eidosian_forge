from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1AnnotationSpec(_messages.Message):
    """Identifies a concept with which DataItems may be annotated with.

  Fields:
    createTime: Output only. Timestamp when this AnnotationSpec was created.
    displayName: Required. The user-defined name of the AnnotationSpec. The
      name can be up to 128 characters long and can consist of any UTF-8
      characters.
    etag: Optional. Used to perform consistent read-modify-write updates. If
      not set, a blind "overwrite" update happens.
    name: Output only. Resource name of the AnnotationSpec.
    updateTime: Output only. Timestamp when AnnotationSpec was last updated.
  """
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    etag = _messages.StringField(3)
    name = _messages.StringField(4)
    updateTime = _messages.StringField(5)