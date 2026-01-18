from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1p3beta1StreamingAnnotateVideoResponse(_messages.Message):
    """`StreamingAnnotateVideoResponse` is the only message returned to the
  client by `StreamingAnnotateVideo`. A series of zero or more
  `StreamingAnnotateVideoResponse` messages are streamed back to the client.

  Fields:
    annotationResults: Streaming annotation results.
    annotationResultsUri: Google Cloud Storage URI that stores annotation
      results of one streaming session in JSON format. It is the
      annotation_result_storage_directory from the request followed by
      '/cloud_project_number-session_id'.
    error: If set, returns a google.rpc.Status message that specifies the
      error for the operation.
  """
    annotationResults = _messages.MessageField('GoogleCloudVideointelligenceV1p3beta1StreamingVideoAnnotationResults', 1)
    annotationResultsUri = _messages.StringField(2)
    error = _messages.MessageField('GoogleRpcStatus', 3)