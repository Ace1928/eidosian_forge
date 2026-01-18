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
def AddRBACResourceArg(parser, api_version='v1', rbacrb_help=''):
    """Add resource arg for projects/{}/locations/{}/memberships/{}."""
    flag_name = 'NAME'
    spec = concepts.ResourceSpec('gkehub.projects.locations.namespaces.rbacrolebindings', api_version=api_version, resource_name='rbacrolebinding', plural_name='rbacrolebindings', disable_auto_completers=True, projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=_LocationAttributeConfig(), namespacesId=_BasicAttributeConfig('namespace', ''), rbacrolebindingsId=_BasicAttributeConfig('rbacrolebinding', rbacrb_help))
    concept_parsers.ConceptParser.ForResource(flag_name, spec, 'The group of arguments defining an RBACRoleBinding.', plural=False, required=True).AddToParser(parser)