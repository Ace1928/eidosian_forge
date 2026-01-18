from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1UploadedNpmPackage(_messages.Message):
    """An npm package uploaded to Artifact Registry using the NpmPackage
  directive.

  Fields:
    fileHashes: Hash types and values of the npm package.
    pushTiming: Output only. Stores timing information for pushing the
      specified artifact.
    uri: URI of the uploaded npm package.
  """
    fileHashes = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1FileHashes', 1)
    pushTiming = _messages.MessageField('ContaineranalysisGoogleDevtoolsCloudbuildV1TimeSpan', 2)
    uri = _messages.StringField(3)