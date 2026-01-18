from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemoteworkersV1test2Blob(_messages.Message):
    """Describes a blob of binary content with its digest.

  Fields:
    contents: The contents of the blob.
    digest: The digest of the blob. This should be verified by the receiver.
  """
    contents = _messages.BytesField(1)
    digest = _messages.MessageField('GoogleDevtoolsRemoteworkersV1test2Digest', 2)