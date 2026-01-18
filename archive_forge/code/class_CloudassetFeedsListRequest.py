from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetFeedsListRequest(_messages.Message):
    """A CloudassetFeedsListRequest object.

  Fields:
    parent: Required. The parent project/folder/organization whose feeds are
      to be listed. It can only be using project/folder/organization number
      (such as "folders/12345")", or a project ID (such as "projects/my-
      project-id").
  """
    parent = _messages.StringField(1, required=True)