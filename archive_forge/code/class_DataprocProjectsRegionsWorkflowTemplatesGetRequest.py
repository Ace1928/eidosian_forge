from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsWorkflowTemplatesGetRequest(_messages.Message):
    """A DataprocProjectsRegionsWorkflowTemplatesGetRequest object.

  Fields:
    name: Required. The resource name of the workflow template, as described
      in https://cloud.google.com/apis/design/resource_names. For
      projects.regions.workflowTemplates.get, the resource name of the
      template has the following format:
      projects/{project_id}/regions/{region}/workflowTemplates/{template_id}
      For projects.locations.workflowTemplates.get, the resource name of the
      template has the following format: projects/{project_id}/locations/{loca
      tion}/workflowTemplates/{template_id}
    version: Optional. The version of workflow template to retrieve. Only
      previously instantiated versions can be retrieved.If unspecified,
      retrieves the current version.
  """
    name = _messages.StringField(1, required=True)
    version = _messages.IntegerField(2, variant=_messages.Variant.INT32)