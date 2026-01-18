from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatapipelinesV1LaunchTemplateRequest(_messages.Message):
    """A request to launch a template.

  Fields:
    gcsPath: A Cloud Storage path to the template from which to create the
      job. Must be a valid Cloud Storage URL, beginning with 'gs://'.
    launchParameters: The parameters of the template to launch. This should be
      part of the body of the POST request.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints) to
      which to direct the request.
    projectId: Required. The ID of the Cloud Platform project that the job
      belongs to.
    validateOnly: If true, the request is validated but not actually executed.
      Defaults to false.
  """
    gcsPath = _messages.StringField(1)
    launchParameters = _messages.MessageField('GoogleCloudDatapipelinesV1LaunchTemplateParameters', 2)
    location = _messages.StringField(3)
    projectId = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)