from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.secrets import completers as secrets_completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def _ArgOrFlag(name, positional):
    """Returns the argument name in resource argument format or flag format.

  Args:
      name (str): name of the argument
      positional (bool): whether the argument is positional

  Returns:
      arg (str): the argument or flag
  """
    if positional:
        return name.upper().replace('-', '_')
    return '--{}'.format(name)