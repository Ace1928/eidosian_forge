from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.calliope.exceptions import core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import times
from mako import runtime
from mako import template
def _BuildTemplate(template_file_name):
    dir_name = os.path.dirname(__file__)
    template_path = os.path.join(dir_name, 'terraform_templates', template_file_name)
    file_template = template.Template(filename=template_path)
    return file_template