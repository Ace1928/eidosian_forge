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
def GetMultiScopeListerHelp(resource, scopes):
    """Returns the detailed help dict for a global and regional list command."""
    zone_example_text = '\nTo list all {0} in zones ``us-central1-b\'\'\nand ``europe-west1-d\'\', given they are zonal resources, run:\n\n  $ {{command}} --filter="zone:( europe-west1-d us-central1-b )"\n'
    region_example_text = '\nTo list all {0} in the ``us-central1\'\' and ``europe-west1\'\' regions,\ngiven they are regional resources, run:\n\n  $ {{command}} --filter="region:( europe-west1 us-central1 )"\n'
    global_example_text = '\nTo list all global {0} in a project, run:\n\n  $ {{command}} --global\n'
    allowed_flags = []
    default_result = []
    if ScopeType.global_scope in scopes:
        allowed_flags.append("``--global''")
        default_result.append('global ' + resource)
    if ScopeType.regional_scope in scopes:
        allowed_flags.append("``--regions''")
        default_result.append(resource + ' from all regions')
    if ScopeType.zonal_scope in scopes:
        allowed_flags.append("``--zones''")
        default_result.append(resource + ' from all zones')
    allowed_flags_text = ', '.join(allowed_flags[:-1]) + ' or ' + allowed_flags[-1]
    default_result_text = ', '.join(default_result[:-1]) + ' and ' + default_result[-1]
    return {'brief': 'List Google Compute Engine ' + resource, 'DESCRIPTION': '\n*{{command}}* displays all Google Compute Engine {0} in a project.\n\nBy default, {1} are listed. The results can be narrowed down by\nproviding the {2} flag.\n'.format(resource, default_result_text, allowed_flags_text), 'EXAMPLES': ('\nTo list all {0} in a project in table form, run:\n\n  $ {{command}}\n\nTo list the URIs of all {0} in a project, run:\n\n  $ {{command}} --uri\n' + (global_example_text if ScopeType.global_scope in scopes else '') + (region_example_text if ScopeType.regional_scope in scopes else '') + (zone_example_text if ScopeType.zonal_scope in scopes else '')).format(resource)}