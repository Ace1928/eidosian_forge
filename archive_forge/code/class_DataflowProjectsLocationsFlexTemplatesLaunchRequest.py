from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsFlexTemplatesLaunchRequest(_messages.Message):
    """A DataflowProjectsLocationsFlexTemplatesLaunchRequest object.

  Fields:
    launchFlexTemplateRequest: A LaunchFlexTemplateRequest resource to be
      passed as the request body.
    location: Required. The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints) to
      which to direct the request. E.g., us-central1, us-west1.
    projectId: Required. The ID of the Cloud Platform project that the job
      belongs to.
  """
    launchFlexTemplateRequest = _messages.MessageField('LaunchFlexTemplateRequest', 1)
    location = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)