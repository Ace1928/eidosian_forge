from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SlsaProvenanceV1(_messages.Message):
    """Keep in sync with schema at https://github.com/slsa-
  framework/slsa/blob/main/docs/provenance/schema/v1/provenance.proto Builder
  renamed to ProvenanceBuilder because of Java conflicts.

  Fields:
    buildDefinition: A BuildDefinition attribute.
    runDetails: A RunDetails attribute.
  """
    buildDefinition = _messages.MessageField('BuildDefinition', 1)
    runDetails = _messages.MessageField('RunDetails', 2)