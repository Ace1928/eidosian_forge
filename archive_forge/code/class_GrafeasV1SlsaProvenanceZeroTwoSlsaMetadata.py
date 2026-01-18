from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrafeasV1SlsaProvenanceZeroTwoSlsaMetadata(_messages.Message):
    """Other properties of the build.

  Fields:
    buildFinishedOn: A string attribute.
    buildInvocationId: A string attribute.
    buildStartedOn: A string attribute.
    completeness: A GrafeasV1SlsaProvenanceZeroTwoSlsaCompleteness attribute.
    reproducible: A boolean attribute.
  """
    buildFinishedOn = _messages.StringField(1)
    buildInvocationId = _messages.StringField(2)
    buildStartedOn = _messages.StringField(3)
    completeness = _messages.MessageField('GrafeasV1SlsaProvenanceZeroTwoSlsaCompleteness', 4)
    reproducible = _messages.BooleanField(5)