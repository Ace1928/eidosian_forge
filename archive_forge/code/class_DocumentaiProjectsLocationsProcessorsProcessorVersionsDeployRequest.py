from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsProcessorVersionsDeployRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorsProcessorVersionsDeployRequest
  object.

  Fields:
    googleCloudDocumentaiV1DeployProcessorVersionRequest: A
      GoogleCloudDocumentaiV1DeployProcessorVersionRequest resource to be
      passed as the request body.
    name: Required. The processor version resource name to be deployed.
  """
    googleCloudDocumentaiV1DeployProcessorVersionRequest = _messages.MessageField('GoogleCloudDocumentaiV1DeployProcessorVersionRequest', 1)
    name = _messages.StringField(2, required=True)