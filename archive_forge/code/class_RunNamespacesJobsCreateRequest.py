from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesJobsCreateRequest(_messages.Message):
    """A RunNamespacesJobsCreateRequest object.

  Fields:
    job: A Job resource to be passed as the request body.
    parent: Required. The namespace in which the job should be created.
      Replace {namespace} with the project ID or number. It takes the form
      namespaces/{namespace}. For example: namespaces/PROJECT_ID
  """
    job = _messages.MessageField('Job', 1)
    parent = _messages.StringField(2, required=True)