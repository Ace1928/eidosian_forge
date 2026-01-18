from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1alpha1UploadAptArtifactResponse(_messages.Message):
    """The response of the completed artifact upload operation. This response
  is contained in the Operation and available to users.

  Fields:
    aptArtifacts: The Apt artifacts updated.
  """
    aptArtifacts = _messages.MessageField('GoogleDevtoolsArtifactregistryV1alpha1AptArtifact', 1, repeated=True)