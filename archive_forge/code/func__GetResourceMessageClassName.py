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
def _GetResourceMessageClassName(singular_name):
    """Returns the properly capitalized resource class name."""
    resource_name = singular_name.strip()
    if len(resource_name) > 1:
        return resource_name[0].upper() + resource_name[1:]
    return resource_name.capitalize()