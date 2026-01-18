from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.deployment_manager import importer
from googlecloudsdk.core import properties
Build a TemplateContents message from a local template or url.

  Args:
    messages: The API message to use.
    template_path: Path to the config yaml file, with an optional list of
      imports.

  Returns:
    The TemplateContents message from the template at template_path.

  Raises:
    Error if the provided file is not a template.
  