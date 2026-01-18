from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunDetails(_messages.Message):
    """A RunDetails object.

  Fields:
    builder: A ProvenanceBuilder attribute.
    byproducts: A ResourceDescriptor attribute.
    metadata: A BuildMetadata attribute.
  """
    builder = _messages.MessageField('ProvenanceBuilder', 1)
    byproducts = _messages.MessageField('ResourceDescriptor', 2, repeated=True)
    metadata = _messages.MessageField('BuildMetadata', 3)