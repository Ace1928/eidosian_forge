from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1alpha1GoogetArtifact(_messages.Message):
    """A detailed representation of a GooGet artifact.

  Fields:
    architecture: Output only. Operating system architecture of the artifact.
    name: Output only. The Artifact Registry resource name of the artifact.
    packageName: Output only. The GooGet package name of the artifact.
  """
    architecture = _messages.StringField(1)
    name = _messages.StringField(2)
    packageName = _messages.StringField(3)