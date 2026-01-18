from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootHarmSpiiFilter(_messages.Message):
    """A LearningGenaiRootHarmSpiiFilter object.

  Fields:
    usBankRoutingMicr: A boolean attribute.
    usEmployerIdentificationNumber: A boolean attribute.
    usSocialSecurityNumber: A boolean attribute.
  """
    usBankRoutingMicr = _messages.BooleanField(1)
    usEmployerIdentificationNumber = _messages.BooleanField(2)
    usSocialSecurityNumber = _messages.BooleanField(3)