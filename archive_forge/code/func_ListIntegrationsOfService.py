from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
def ListIntegrationsOfService(parser):
    """Filter by Service Name."""
    parser.add_argument('--service', type=str, help='Filter Integrations by Name of Cloud Run service.')