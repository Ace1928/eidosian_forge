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
def CheckArgs(self, args):
    if not args.add_single_attribute_selectors and (not args.remove_single_attribute_selectors) and (not args.single_attribute_selectors):
        raise gcloud_exceptions.OneOfArgumentsRequiredException(['--single-attribute-selectors', '--add-single-attribute-selectors', '--remove-attribute-selectors'], 'Must add or remove at least one selector that will match workload(s) from the source.')