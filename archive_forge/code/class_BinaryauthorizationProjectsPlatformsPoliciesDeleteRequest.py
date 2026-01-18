from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BinaryauthorizationProjectsPlatformsPoliciesDeleteRequest(_messages.Message):
    """A BinaryauthorizationProjectsPlatformsPoliciesDeleteRequest object.

  Fields:
    name: Required. The name of the platform policy to delete, in the format
      `projects/*/platforms/*/policies/*`.
  """
    name = _messages.StringField(1, required=True)