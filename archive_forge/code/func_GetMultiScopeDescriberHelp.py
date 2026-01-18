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
def GetMultiScopeDescriberHelp(resource, scopes):
    """Returns the detailed help dict for a multiscope describe command.

  Args:
    resource: resource name, singular form with no preposition
    scopes: global/regional/zonal or mix of them

  Returns:
    Help for multi-scope describe command.
  """
    article = text.GetArticle(resource)
    zone_example_text = "To get details about a zonal {0} in the ``us-central1-b'' zone, run:\n\n  $ {{command}} --zone=us-central1-b\n\n"
    region_example_text = "To get details about a regional {0} in the ``us-central1'' regions, run:\n\n  $ {{command}} --region=us-central1\n\n"
    global_example_text = 'To get details about a global {0}, run:\n\n  $ {{command}} --global\n\n'
    return {'brief': 'Display detailed information about {0} {1}'.format(article, resource), 'DESCRIPTION': '\n*{{command}}* displays all data associated with {0} {1} in a project.\n'.format(article, resource), 'EXAMPLES': ((global_example_text if ScopeType.global_scope in scopes else '') + (region_example_text if ScopeType.regional_scope in scopes else '') + (zone_example_text if ScopeType.zonal_scope in scopes else '')).format(resource)}