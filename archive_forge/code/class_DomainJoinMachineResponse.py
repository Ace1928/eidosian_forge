from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainJoinMachineResponse(_messages.Message):
    """DomainJoinMachineResponse is the response message for DomainJoinMachine
  method

  Fields:
    domainJoinBlob: Offline domain join blob as the response
  """
    domainJoinBlob = _messages.StringField(1)