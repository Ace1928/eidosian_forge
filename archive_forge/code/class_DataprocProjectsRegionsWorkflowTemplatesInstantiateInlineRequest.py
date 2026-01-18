from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsWorkflowTemplatesInstantiateInlineRequest(_messages.Message):
    """A DataprocProjectsRegionsWorkflowTemplatesInstantiateInlineRequest
  object.

  Fields:
    parent: Required. The resource name of the region or location, as
      described in https://cloud.google.com/apis/design/resource_names. For
      projects.regions.workflowTemplates,instantiateinline, the resource name
      of the region has the following format:
      projects/{project_id}/regions/{region} For
      projects.locations.workflowTemplates.instantiateinline, the resource
      name of the location has the following format:
      projects/{project_id}/locations/{location}
    requestId: Optional. A tag that prevents multiple concurrent workflow
      instances with the same tag from running. This mitigates risk of
      concurrent instances started due to retries.It is recommended to always
      set this value to a UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier).The tag
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
    workflowTemplate: A WorkflowTemplate resource to be passed as the request
      body.
  """
    parent = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    workflowTemplate = _messages.MessageField('WorkflowTemplate', 3)