from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BinaryauthorizationProjectsAttestorsDeleteRequest(_messages.Message):
    """A BinaryauthorizationProjectsAttestorsDeleteRequest object.

  Fields:
    name: Required. The name of the attestors to delete, in the format
      `projects/*/attestors/*`.
  """
    name = _messages.StringField(1, required=True)