from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityAssignment(_messages.Message):
    """Defines how to assign an identity to a workload. At least one workload
  selector and at least one identity assignment method must be defined.

  Fields:
    allowIdentitySelfSelection: Optional. Identity assignment method that
      authorizes matched workloads to self select an identity within the
      parent's scope (e.g. within the namespace when the WorkloadSource is
      defined on a Namespace).
    singleAttributeSelectors: Optional. Workload selector that matches
      workloads based on their attested attributes.
  """
    allowIdentitySelfSelection = _messages.BooleanField(1)
    singleAttributeSelectors = _messages.MessageField('SingleAttributeSelector', 2, repeated=True)