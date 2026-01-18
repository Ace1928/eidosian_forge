from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StatefulPolicy(_messages.Message):
    """A StatefulPolicy object.

  Fields:
    preservedState: A StatefulPolicyPreservedState attribute.
  """
    preservedState = _messages.MessageField('StatefulPolicyPreservedState', 1)