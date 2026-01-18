from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
Gets workflow template from dataproc.

    Args:
      template: workflow template resource that contains template name and id.
      version: version of the workflow template to get.

    Returns:
      WorkflowTemplate object that contains the workflow template info.

    Raises:
      ValueError: if version cannot be converted to a valid integer.
    