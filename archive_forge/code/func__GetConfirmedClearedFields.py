from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import instances as command_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def _GetConfirmedClearedFields(args, patch_instance, original_instance):
    """Clear fields according to args and confirm with user."""
    cleared_fields = []
    if args.clear_gae_apps:
        cleared_fields.append('settings.authorizedGaeApplications')
    if args.clear_authorized_networks:
        cleared_fields.append('settings.ipConfiguration.authorizedNetworks')
    if args.clear_database_flags:
        cleared_fields.append('settings.databaseFlags')
    if args.remove_deny_maintenance_period:
        cleared_fields.append('settings.denyMaintenancePeriods')
    if args.clear_password_policy:
        cleared_fields.append('settings.passwordValidationPolicy')
    if args.IsKnownAndSpecified('clear_allowed_psc_projects'):
        cleared_fields.append('settings.ipConfiguration.pscConfig.allowedConsumerProjects')
    log.status.write('The following message will be used for the patch API method.\n')
    log.status.write(encoding.MessageToJson(WithoutKind(patch_instance), include_fields=cleared_fields) + '\n')
    _PrintAndConfirmWarningMessage(args, original_instance.databaseVersion)
    return cleared_fields