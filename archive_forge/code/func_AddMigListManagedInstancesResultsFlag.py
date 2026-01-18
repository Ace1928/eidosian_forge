from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from typing import Any
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddMigListManagedInstancesResultsFlag(parser):
    """Add --list-managed-instances-results flag to the parser."""
    help_text = "      Pagination behavior for the group's listManagedInstances API method.\n      This flag does not affect the group's gcloud or console list-instances\n      behavior. By default it is set to ``pageless''.\n    "
    choices = {'pageless': "Pagination is disabled for the group's listManagedInstances API method. maxResults and pageToken query parameters are ignored and all instances are returned in a single response.", 'paginated': "Pagination is enabled for the group's listManagedInstances API method. maxResults and pageToken query parameters are respected."}
    parser.add_argument('--list-managed-instances-results', metavar='MODE', type=lambda x: x.lower(), choices=choices, help=help_text)