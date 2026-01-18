from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def GetRegionsWorkflowTemplate(self, template, version=None):
    """Gets workflow template from dataproc.

    Args:
      template: workflow template resource that contains template name and id.
      version: version of the workflow template to get.

    Returns:
      WorkflowTemplate object that contains the workflow template info.

    Raises:
      ValueError: if version cannot be converted to a valid integer.
    """
    messages = self.messages
    get_request = messages.DataprocProjectsRegionsWorkflowTemplatesGetRequest(name=template.RelativeName())
    if version:
        get_request.version = int(version)
    return self.client.projects_regions_workflowTemplates.Get(get_request)