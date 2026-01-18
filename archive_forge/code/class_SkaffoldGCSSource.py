from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SkaffoldGCSSource(_messages.Message):
    """Cloud Storage bucket containing Skaffold Config modules.

  Fields:
    path: Optional. Relative path from the source to the Skaffold file.
    source: Required. Cloud Storage source paths to copy recursively. For
      example, providing "gs://my-bucket/dir/configs/*" will result in
      Skaffold copying all files within the "dir/configs" directory in the
      bucket "my-bucket".
  """
    path = _messages.StringField(1)
    source = _messages.StringField(2)