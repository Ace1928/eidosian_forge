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
class BaseLister(base.ListCommand, BaseCommand):
    """Base class for the list subcommands."""
    self_links = None
    names = None
    resource_refs = None
    service = None

    @staticmethod
    def Args(parser):
        parser.add_argument('names', metavar='NAME', nargs='*', default=[], completer=completers.InstancesCompleter, help='If provided, show details for the specified names and/or URIs of resources.')
        parser.add_argument('--regexp', '-r', help='        Regular expression to filter the names of the results on. Any names\n        that do not match the entire regular expression will be filtered out.\n        ')

    @property
    def allowed_filtering_types(self):
        """The list of resource types that can be provided to filtering."""
        return [self.resource_type]

    @abc.abstractmethod
    def GetResources(self, args, errors):
        """Returns a generator of JSON-serializable resource dicts."""

    def GetFilterExpr(self, args):
        """Returns a filter expression if --regexp is provided."""
        if args.regexp:
            return 'name eq {0}'.format(args.regexp)
        else:
            return None

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

    def FilterResults(self, args, items):
        """Filters the list results by name and URI."""
        for item in items:
            if not args.names:
                yield item
            elif item['selfLink'] in self.self_links or item['name'] in self.names:
                yield item

    def ComputeDynamicProperties(self, args, items):
        """Computes dynamic properties, which are not returned by GCE API."""
        _ = args
        return items

    def Run(self, args):
        if not args.IsSpecified('format') and (not args.uri) and self.Collection():
            r = resource_registry.RESOURCE_REGISTRY[self.Collection()]
            args.format = r.list_format
        return self._Run(args)

    def _Run(self, args):
        """Yields JSON-serializable dicts of resources or self links."""
        self.self_links = set()
        self.names = set()
        self.resource_refs = []
        field_selector = property_selector.PropertySelector(properties=None, transformations=self.transformations)
        errors = []
        self.PopulateResourceFilteringStructures(args)
        items = self.FilterResults(args, self.GetResources(args, errors))
        items = lister.ProcessResults(resources=items, field_selector=field_selector)
        items = self.ComputeDynamicProperties(args, items)
        for item in items:
            yield item
        if errors:
            utils.RaiseToolException(errors)