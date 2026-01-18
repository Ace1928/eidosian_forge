from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BinaryauthorizationSystempolicyGetPolicyRequest(_messages.Message):
    """A BinaryauthorizationSystempolicyGetPolicyRequest object.

  Fields:
    name: Required. The resource name, in the format `locations/*/policy`.
      Note that the system policy is not associated with a project.
  """
    name = _messages.StringField(1, required=True)