from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InToto(_messages.Message):
    """This contains the fields corresponding to the definition of a software
  supply chain step in an in-toto layout. This information goes into a Grafeas
  note.

  Fields:
    expectedCommand: This field contains the expected command used to perform
      the step.
    expectedMaterials: The following fields contain in-toto artifact rules
      identifying the artifacts that enter this supply chain step, and exit
      the supply chain step, i.e. materials and products of the step.
    expectedProducts: A ArtifactRule attribute.
    signingKeys: This field contains the public keys that can be used to
      verify the signatures on the step metadata.
    stepName: This field identifies the name of the step in the supply chain.
    threshold: This field contains a value that indicates the minimum number
      of keys that need to be used to sign the step's in-toto link.
  """
    expectedCommand = _messages.StringField(1, repeated=True)
    expectedMaterials = _messages.MessageField('ArtifactRule', 2, repeated=True)
    expectedProducts = _messages.MessageField('ArtifactRule', 3, repeated=True)
    signingKeys = _messages.MessageField('SigningKey', 4, repeated=True)
    stepName = _messages.StringField(5)
    threshold = _messages.IntegerField(6)