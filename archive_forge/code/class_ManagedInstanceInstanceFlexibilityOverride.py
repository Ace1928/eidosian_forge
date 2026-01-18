from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedInstanceInstanceFlexibilityOverride(_messages.Message):
    """A ManagedInstanceInstanceFlexibilityOverride object.

  Fields:
    machineType: The machine type to be used for this instance.
  """
    machineType = _messages.StringField(1)