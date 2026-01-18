from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsWorkflowTemplatesCreateRequest(_messages.Message):
    """A DataprocProjectsRegionsWorkflowTemplatesCreateRequest object.

  Fields:
    parent: Required. The resource name of the region or location, as
      described in https://cloud.google.com/apis/design/resource_names. For
      projects.regions.workflowTemplates.create, the resource name of the
      region has the following format: projects/{project_id}/regions/{region}
      For projects.locations.workflowTemplates.create, the resource name of
      the location has the following format:
      projects/{project_id}/locations/{location}
    workflowTemplate: A WorkflowTemplate resource to be passed as the request
      body.
  """
    parent = _messages.StringField(1, required=True)
    workflowTemplate = _messages.MessageField('WorkflowTemplate', 2)