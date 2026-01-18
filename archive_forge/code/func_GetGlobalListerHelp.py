from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import argparse  # pylint: disable=unused-import
import json
import textwrap
from apitools.base.py import base_api  # pylint: disable=unused-import
import enum
from googlecloudsdk.api_lib.compute import base_classes_resource_registry as resource_registry
from googlecloudsdk.api_lib.compute import client_adapter
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import property_selector
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import resource_specs
from googlecloudsdk.api_lib.compute import scope_prompter
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import text
import six
def GetGlobalListerHelp(resource):
    """Returns the detailed help dict for a global list command."""
    if resource == 'routes':
        detailed_help = {'brief': 'List non-dynamic Google Compute Engine ' + resource, 'DESCRIPTION': '\n*{{command}}* displays all custom static, subnet, and peering {0} in\nVPC networks in a project.\n\nTo list custom dynamic routes learned by Cloud Routers, query the\nstatus of the Cloud Router that learned the route using\n`gcloud compute routers get-status`. For more details, refer\nto https://cloud.google.com/vpc/docs/using-routes#listingroutes.\n'.format(resource), 'EXAMPLES': '\nTo list all non-dynamic {0} in a project in table form, run:\n\n    $ {{command}}\n\nTo list the URIs of all non-dynamic {0} in a project, run:\n\n    $ {{command}} --uri\n'.format(resource)}
    else:
        detailed_help = {'brief': 'List Google Compute Engine ' + resource, 'DESCRIPTION': '\n*{{command}}* displays all Google Compute Engine {0} in a project.\n'.format(resource), 'EXAMPLES': '\nTo list all {0} in a project in table form, run:\n\n  $ {{command}}\n\nTo list the URIs of all {0} in a project, run:\n\n  $ {{command}} --uri\n'.format(resource)}
    if resource == 'images':
        detailed_help['EXAMPLES'] += '\nTo list the names of {0} older than one year from oldest to newest\n(`-P1Y` is an [ISO8601 duration](https://en.wikipedia.org/wiki/ISO_8601)):\n\n  $ {{command}} --format="value(NAME)" --filter="creationTimestamp < -P1Y"\n'.format(resource)
    return detailed_help