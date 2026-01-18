from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudOrgpolicyV2ConstraintBooleanConstraint(_messages.Message):
    """A constraint that is either enforced or not. For example, a constraint
  `constraints/compute.disableSerialPortAccess`. If it is enforced on a VM
  instance, serial port connections will not be opened to that instance.
  """