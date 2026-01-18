from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import update_args
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import resources
def GetArgFromNamespace(self, namespace, arg):
    """Retrieves namespace value associated with flag.

    Args:
      namespace: The parsed command line argument namespace.
      arg: base.Argument|concept_parsers.ConceptParser|None, used to get
        namespace value

    Returns:
      value parsed from namespace
    """
    if isinstance(arg, base.Argument):
        return arg_utils.GetFromNamespace(namespace, arg.name)
    if isinstance(arg, concept_parsers.ConceptParser):
        all_anchors = list(arg.specs.keys())
        if len(all_anchors) != 1:
            raise ValueError('ConceptParser must contain exactly one spec for clearable but found specs {}. {} cannot parse the namespace value if more than or less than one spec is added to the ConceptParser.'.format(all_anchors, type(self).__name__))
        name = all_anchors[0]
        value = arg_utils.GetFromNamespace(namespace.CONCEPTS, name)
        if value:
            value = value.Parse()
        return value
    return None