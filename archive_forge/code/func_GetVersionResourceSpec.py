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
def GetVersionResourceSpec():
    return concepts.ResourceSpec('secretmanager.projects.secrets.versions', resource_name='version', plural_name='version', disable_auto_completers=False, versionsId=GetVersionAttributeConfig(), secretsId=GetSecretAttributeConfig(), projectsId=GetProjectAttributeConfig())