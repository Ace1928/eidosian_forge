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
def AddMembershipResourceArg(parser, api_version='v1', positional=False, plural=False, membership_required=False, flag_override='', membership_help='', location_help=''):
    """Add resource arg for projects/{}/locations/{}/memberships/{}."""
    flag_name = '--membership'
    if flag_override:
        flag_name = flag_override
    elif positional:
        flag_name = 'MEMBERSHIP_NAME'
    elif plural:
        flag_name = '--memberships'
    spec = concepts.ResourceSpec('gkehub.projects.locations.memberships', api_version=api_version, resource_name='membership', plural_name='memberships', disable_auto_completers=True, projectsId=concepts.DEFAULT_PROJECT_ATTRIBUTE_CONFIG, locationsId=_LocationAttributeConfig(location_help), membershipsId=_BasicAttributeConfig('memberships' if plural else 'membership', membership_help))
    concept_parsers.ConceptParser.ForResource(flag_name, spec, 'The group of arguments defining one or more memberships.' if plural else 'The group of arguments defining a membership.', plural=plural, required=membership_required).AddToParser(parser)