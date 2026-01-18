from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExtendSchemaRequest(_messages.Message):
    """ExtendSchemaRequest is the request message for ExtendSchema method.

  Fields:
    description: Required. Description for Schema Change.
    fileContents: File uploaded as a byte stream input.
    gcsPath: File stored in Cloud Storage bucket and represented in the form
      projects/{project_id}/buckets/{bucket_name}/objects/{object_name} File
      should be in the same project as the domain.
  """
    description = _messages.StringField(1)
    fileContents = _messages.BytesField(2)
    gcsPath = _messages.StringField(3)