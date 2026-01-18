from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os.path
import re
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def _TemplateFileForCommandPath(command_template_filename, test=False):
    """Returns Mako template corresping to command_template_filename.

  Args:
    command_template_filename: name of file containing template (no path).
    test: if the template file should be a test file, defaults to False.
  """
    if test:
        template_dir = 'test_templates'
    else:
        template_dir = 'command_templates'
    template_path = os.path.join(os.path.dirname(__file__), template_dir, command_template_filename)
    return template.Template(filename=template_path)