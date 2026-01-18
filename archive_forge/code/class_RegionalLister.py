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
class RegionalLister(BaseLister):
    """Base class for listing regional resources."""

    @staticmethod
    def Args(parser):
        BaseLister.Args(parser)
        parser.add_argument('--regions', metavar='REGION', help='If provided, only resources from the given regions are queried.', type=arg_parsers.ArgList(min_length=1), default=[])

    def GetResources(self, args, errors):
        region_names = [self.CreateGlobalReference(region, resource_type='regions').Name() for region in args.regions]
        return lister.GetRegionalResourcesDicts(service=self.service, project=self.project, requested_regions=region_names, filter_expr=self.GetFilterExpr(args), http=self.http, batch_url=self.batch_url, errors=errors)