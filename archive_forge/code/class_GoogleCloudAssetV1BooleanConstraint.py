from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1BooleanConstraint(_messages.Message):
    """A `Constraint` that is either enforced or not. For example a constraint
  `constraints/compute.disableSerialPortAccess`. If it is enforced on a VM
  instance, serial port connections will not be opened to that instance.
  """