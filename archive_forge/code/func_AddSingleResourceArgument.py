from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddSingleResourceArgument(parser, resource_path, help_text, fallthroughs=tuple(), positional=True, argument_name=None, required=None, prefixes=False, validate=False, help_texts=None):
    """Creates a concept parser for `resource_path` and adds it to `parser`.

  Args:
    parser: the argparse.ArgumentParser to which the concept parser will be
      added.
    resource_path: path to the resource, in `entity.other_entity.leaf` format.
    help_text: the help text to display when describing the resource as a whole.
    fallthroughs: fallthrough providers for entities in resource_path.
    positional: whether the leaf entity should be provided as a positional
      argument, rather than as a flag.
    argument_name: what to name the leaf entity argument. Defaults to the leaf
      entity name from the resource path.
    required: whether the user is required to provide this resource. Defaults to
      True for positional arguments, False otherwise.
    prefixes: whether to append prefixes to the non-leaf arguments.
    validate: whether to check that the user-provided resource matches the
      expected naming conventions of the resource path.
    help_texts: custom help text for generated arguments. Defaults to each
      entity using a generic help text.
  """
    split_path = resource_path.split('.')
    if argument_name is None:
        leaf_element_name = split_path[-1]
        if positional:
            argument_name = leaf_element_name.upper()
        else:
            argument_name = '--' + leaf_element_name.replace('_', '-')
    if required is None:
        required = positional
    concept_parsers.ConceptParser.ForResource(argument_name, ResourceSpec(split_path, fallthroughs, help_texts, validate=validate), help_text, required=required, prefixes=prefixes).AddToParser(parser)