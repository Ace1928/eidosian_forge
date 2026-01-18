from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipRuntimeStatus(_messages.Message):
    """MembershipRuntimeStatus contains aggregate data about policy controller
  resources on a cluster that is a member of a fleet.

  Fields:
    numConstraintTemplates: The number of constraint templates on the member
      cluster.
    numConstraintViolations: The number of constraint violations on the member
      cluster.
    numConstraints: The number of constraints on the member cluster.
  """
    numConstraintTemplates = _messages.IntegerField(1)
    numConstraintViolations = _messages.IntegerField(2)
    numConstraints = _messages.IntegerField(3)