from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.iam import identity_pool_waiter
from googlecloudsdk.command_lib.iam.workforce_pools import flags
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateAccessRestrictions(self, args, messages):
    if args.IsSpecified('allowed_services') or args.IsSpecified('disable_programmatic_signin'):
        access_restrictions = messages.AccessRestrictions()
        if args.IsSpecified('allowed_services'):
            access_restrictions.allowedServices = args.allowed_services
        if args.IsSpecified('disable_programmatic_signin'):
            access_restrictions.disableProgrammaticSignin = args.disable_programmatic_signin
        return access_restrictions
    return None