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
def PopulateResourceFilteringStructures(self, args):
    """Processes the positional arguments for later filtering."""
    allowed_collections = ['compute.{0}'.format(resource_type) for resource_type in self.allowed_filtering_types]
    for name in args.names:
        try:
            ref = self.resources.Parse(name)
            if ref.Collection() not in allowed_collections:
                raise compute_exceptions.InvalidResourceError('Resource URI must be of type {0}. Received [{1}].'.format(' or '.join(('[{0}]'.format(collection) for collection in allowed_collections)), ref.Collection()))
            self.self_links.add(ref.SelfLink())
            self.resource_refs.append(ref)
            continue
        except resources.UserError:
            pass
        self.names.add(name)