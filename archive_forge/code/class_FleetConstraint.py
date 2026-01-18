from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FleetConstraint(_messages.Message):
    """The fleet-wide info for a constraint. Includes the number of constraints
  to avoid clients making serial round trips.

  Fields:
    numMemberships: The number of memberships where this constraint was found.
    numViolations: The number of violations of this constraint.
    ref: The constraint this data refers to.
  """
    numMemberships = _messages.IntegerField(1)
    numViolations = _messages.IntegerField(2)
    ref = _messages.MessageField('ConstraintRef', 3)