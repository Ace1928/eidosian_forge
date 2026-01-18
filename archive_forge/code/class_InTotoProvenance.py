from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InTotoProvenance(_messages.Message):
    """A InTotoProvenance object.

  Fields:
    builderConfig: required
    materials: The collection of artifacts that influenced the build including
      sources, dependencies, build tools, base images, and so on. This is
      considered to be incomplete unless metadata.completeness.materials is
      true. Unset or null is equivalent to empty.
    metadata: A Metadata attribute.
    recipe: Identifies the configuration used for the build. When combined
      with materials, this SHOULD fully describe the build, such that re-
      running this recipe results in bit-for-bit identical output (if the
      build is reproducible). required
  """
    builderConfig = _messages.MessageField('BuilderConfig', 1)
    materials = _messages.StringField(2, repeated=True)
    metadata = _messages.MessageField('Metadata', 3)
    recipe = _messages.MessageField('Recipe', 4)