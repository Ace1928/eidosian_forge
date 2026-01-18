from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import operator
from googlecloudsdk.api_lib.iam import util
from googlecloudsdk.api_lib.iam.workload_identity_pools import workload_sources
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.iam.workload_identity_pools import flags
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import log
def ReconcileSingleAttributeSelectorList(self, args, original_selector_list):
    updated_selectors = set()
    updated_selectors.update(map(self.ToHashableSelector, original_selector_list))
    if args.add_single_attribute_selectors:
        updated_selectors.update(map(self.ToHashableSelector, flags.ParseSingleAttributeSelectorArg('--add-single-attribute-selectors', args.add_single_attribute_selectors)))
    if args.remove_single_attribute_selectors:
        for selector in map(self.ToHashableSelector, flags.ParseSingleAttributeSelectorArg('--remove-single-attribute-selectors', args.remove_single_attribute_selectors)):
            updated_selectors.discard(selector)
    return sorted(list(map(self.ToProtoSelector, updated_selectors)), key=operator.attrgetter('attribute', 'value'))