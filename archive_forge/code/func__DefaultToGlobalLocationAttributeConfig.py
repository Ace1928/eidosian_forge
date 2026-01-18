from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import util as cmd_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def _DefaultToGlobalLocationAttributeConfig(help_text=''):
    """Create basic attributes that fallthrough location to global in resource argument.

  Args:
    help_text: If set, overrides default help text

  Returns:
    Resource argument parameter config
  """
    return concepts.ResourceParameterAttributeConfig(name='location', fallthroughs=[deps.Fallthrough(function=cmd_util.DefaultToGlobal, hint='global is the only supported location')], help_text=help_text if help_text else 'Name of the {resource}.')