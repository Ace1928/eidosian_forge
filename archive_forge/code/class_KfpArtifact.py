from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KfpArtifact(_messages.Message):
    """A detailed representation of a KFP artifact.

  Fields:
    name: Output only. Resource name of the KFP artifact. Since users don't
      directly interact with this resource, the name will be derived from the
      associated version. For example, when version =
      ".../versions/sha256:abcdef...", the name will be
      ".../kfpArtifacts/sha256:abcdef...".
    version: The version associated with the KFP artifact. Must follow the
      Semantic Versioning standard.
  """
    name = _messages.StringField(1)
    version = _messages.StringField(2)