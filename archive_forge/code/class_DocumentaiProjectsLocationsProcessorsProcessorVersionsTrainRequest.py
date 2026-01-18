from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorsProcessorVersionsTrainRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorsProcessorVersionsTrainRequest
  object.

  Fields:
    googleCloudDocumentaiV1TrainProcessorVersionRequest: A
      GoogleCloudDocumentaiV1TrainProcessorVersionRequest resource to be
      passed as the request body.
    parent: Required. The parent (project, location and processor) to create
      the new version for. Format:
      `projects/{project}/locations/{location}/processors/{processor}`.
  """
    googleCloudDocumentaiV1TrainProcessorVersionRequest = _messages.MessageField('GoogleCloudDocumentaiV1TrainProcessorVersionRequest', 1)
    parent = _messages.StringField(2, required=True)