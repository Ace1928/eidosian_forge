from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def _AddRemoveDenyMaintenancePeriod(group):
    """Adds remove deny maintenance period flag to the group."""
    group.add_argument('--remove-deny-maintenance-period', required=False, hidden=True, action='store_true', default=False, help='Remove the user-specified maintenance deny period.')