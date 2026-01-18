from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrafeasV1SlsaProvenanceZeroTwoSlsaBuilder(_messages.Message):
    """Identifies the entity that executed the recipe, which is trusted to have
  correctly performed the operation and populated this provenance.

  Fields:
    id: A string attribute.
  """
    id = _messages.StringField(1)