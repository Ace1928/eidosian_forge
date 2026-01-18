from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1alpha1ImportYumArtifactsRequest(_messages.Message):
    """The request to import new yum artifacts.

  Fields:
    gcsSource: Google Cloud Storage location where input content is located.
  """
    gcsSource = _messages.MessageField('GoogleDevtoolsArtifactregistryV1alpha1ImportYumArtifactsGcsSource', 1)