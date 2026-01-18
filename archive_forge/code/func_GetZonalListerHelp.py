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
def GetZonalListerHelp(resource):
    """Returns the detailed help dict for a zonal list command."""
    return {'brief': 'List Google Compute Engine ' + resource, 'DESCRIPTION': '\n*{{command}}* displays all Google Compute Engine {0} in a project.\n\nBy default, {0} from all zones are listed. The results can be narrowed\ndown using a filter: `--filter="zone:( ZONE ... )"`.\n'.format(resource), 'EXAMPLES': '\nTo list all {0} in a project in table form, run:\n\n  $ {{command}}\n\nTo list the URIs of all {0} in a project, run:\n\n  $ {{command}} --uri\n\nTo list all {0} in the ``us-central1-b\'\' and ``europe-west1-d\'\' zones,\nrun:\n\n  $ {{command}} --filter="zone:( us-central1-b europe-west1-d )"\n'.format(resource)}