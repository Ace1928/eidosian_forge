from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactRule(_messages.Message):
    """Defines an object to declare an in-toto artifact rule

  Fields:
    artifactRule: A string attribute.
  """
    artifactRule = _messages.StringField(1, repeated=True)