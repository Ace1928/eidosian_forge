from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddAuthConfigArgs(parser, resource_type):
    """Add arguments for auth provider."""
    base.Argument('--audiences', metavar='AUDIENCES', type=arg_parsers.ArgList(), help='List of JWT audiences that are allowed to access a {}.\n\nJWT containing any of these audiences\n(https://tools.ietf.org/html/draft-ietf-oauth-json-web-token-32#section -4.1.3)\nwill be accepted.\n'.format(resource_type)).AddToParser(parser)
    base.Argument('--allowed-issuers', metavar='ALLOWED_ISSUERS', type=arg_parsers.ArgList(), help='List of allowed JWT issuers for a {}.\n\nEach entry must be a valid Google service account, in the following format:\n`service-account-name@project-id.iam.gserviceaccount.com`\n'.format(resource_type)).AddToParser(parser)