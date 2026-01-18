from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.accesscontextmanager import acm_printer
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import levels
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def AddPerimeterUpdateDryRunConfigArgs(parser):
    """Add args for perimeters update-dry-run-config command."""
    update_dry_run_group = parser.add_mutually_exclusive_group()
    _AddClearDryRunConfigArg(update_dry_run_group)
    config_group = update_dry_run_group.add_argument_group()
    _AddResources(config_group, include_set=False)
    _AddRestrictedServices(config_group, include_set=False)
    _AddLevelsUpdate(config_group, include_set=False)
    _AddVpcRestrictionArgs(config_group)