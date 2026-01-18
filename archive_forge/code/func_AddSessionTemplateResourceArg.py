from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import json
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def AddSessionTemplateResourceArg(parser, verb, api_version):
    """Adds a session template resource argument.

  Args:
    parser: The argparse parser for the command.
    verb: The verb to apply to the resource, such as 'to update'.
    api_version: api version, for example v1
  """
    concept_parsers.ConceptParser.ForResource('session_template', _SessionTemplateResourceSpec(api_version), 'The session template to {}.'.format(verb), required=True).AddToParser(parser)