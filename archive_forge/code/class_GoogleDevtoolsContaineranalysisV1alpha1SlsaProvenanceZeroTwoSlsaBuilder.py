from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsContaineranalysisV1alpha1SlsaProvenanceZeroTwoSlsaBuilder(_messages.Message):
    """Identifies the entity that executed the recipe, which is trusted to have
  correctly performed the operation and populated this provenance.

  Fields:
    id: URI indicating the builder's identity.
  """
    id = _messages.StringField(1)