from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def AddServiceAccountAndScopeArgs(parser, instance_exists, extra_scopes_help='', operation='Create', resource='instance'):
    """Add args for configuring service account and scopes.

  This should replace AddScopeArgs (b/30802231).

  Args:
    parser: ArgumentParser, parser to which flags will be added.
    instance_exists: bool, If instance already exists and we are modifying it.
    extra_scopes_help: str, Extra help text for the scopes flag.
    operation: operation being performed, capitalized. E.g. 'Create' or 'Import'
    resource: resource type on which scopes and service account are being added.
      E.g. 'instance' or 'machine image'.
  """
    service_account_group = parser.add_mutually_exclusive_group()
    no_sa_instance_not_exist = '{operation} {resource} without service account'.format(operation=operation, resource=resource)
    service_account_group.add_argument('--no-service-account', action='store_true', help='Remove service account from the {0}'.format(resource) if instance_exists else no_sa_instance_not_exist)
    sa_exists = "You can explicitly specify the Compute Engine default service\n  account using the 'default' alias.\n\n  If not provided, the {0} will use the service account it currently has.\n  ".format(resource)
    sa_not_exists = "\n\n  If not provided, the {0} will use the project's default service account.\n  ".format(resource)
    service_account_help = '  A service account is an identity attached to the {resource}. Its access tokens\n  can be accessed through the instance metadata server and are used to\n  authenticate applications on the instance. The account can be set using an\n  email address corresponding to the required service account. {extra_help}\n  '.format(extra_help=sa_exists if instance_exists else sa_not_exists, resource=resource)
    service_account_group.add_argument('--service-account', help=service_account_help)
    scopes_group = parser.add_mutually_exclusive_group()
    scopes_group.add_argument('--no-scopes', action='store_true', help='Remove all scopes from the {resource}'.format(resource=resource) if instance_exists else '{operation} {resource} without scopes'.format(operation=operation, resource=resource))
    scopes_exists = 'keep the scopes it currently has'
    scopes_not_exists = 'be assigned the default scopes, described below'
    scopes_help = 'If not provided, the {resource} will {exists}. {extra}\n\n{scopes_help}\n'.format(exists=scopes_exists if instance_exists else scopes_not_exists, extra=extra_scopes_help, scopes_help=constants.ScopesHelp(), resource=resource)
    scopes_group.add_argument('--scopes', type=arg_parsers.ArgList(), metavar='SCOPE', help=scopes_help)