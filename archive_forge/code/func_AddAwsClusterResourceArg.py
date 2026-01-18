from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddAwsClusterResourceArg(parser, verb, positional=True):
    """Adds a resource argument for an AWS cluster.

  Args:
    parser: The argparse parser to add the resource arg to.
    verb: str, the verb to describe the resource, such as 'to update'.
    positional: bool, whether the argument is positional or not.
  """
    name = 'cluster' if positional else '--cluster'
    concept_parsers.ConceptParser.ForResource(name, GetAwsClusterResourceSpec(), 'cluster {}.'.format(verb), required=True).AddToParser(parser)