from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesJobsRunRequest(_messages.Message):
    """A RunNamespacesJobsRunRequest object.

  Fields:
    name: Required. The name of the job to run. Replace {namespace} with the
      project ID or number. It takes the form namespaces/{namespace}. For
      example: namespaces/PROJECT_ID
    runJobRequest: A RunJobRequest resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    runJobRequest = _messages.MessageField('RunJobRequest', 2)