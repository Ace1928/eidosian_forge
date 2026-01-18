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
def GetRegionalVersionAttributeConfig():
    """Returns the attribute config for regional secret version."""
    return concepts.ResourceParameterAttributeConfig(name='version', help_text='The version of the {resource}.', completion_request_params={'fieldMask': 'name'}, completion_id_field='name')