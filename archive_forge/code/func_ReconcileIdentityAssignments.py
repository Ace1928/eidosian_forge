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
def ReconcileIdentityAssignments(self, args, original_identity_assignments):
    """Reconciles the identity assignment changes with the original list."""
    _, messages = util.GetClientAndMessages()
    updated_selectors = set()
    for identity_assignment in original_identity_assignments:
        updated_selectors.update(map(self.ToHashableSelector, identity_assignment.singleAttributeSelectors))
    if args.allow_identity_self_selection is not None:
        hashable_selectors = set(map(self.ToHashableSelector, flags.ParseSingleAttributeSelectorArg('--single-attribute-selectors', args.single_attribute_selectors)))
        if args.allow_identity_self_selection:
            updated_selectors |= hashable_selectors
        else:
            updated_selectors -= hashable_selectors
    if updated_selectors == set():
        return []
    identity_assignment_proto = messages.IdentityAssignment()
    identity_assignment_proto.singleAttributeSelectors.extend(sorted(list(map(self.ToProtoSelector, updated_selectors)), key=operator.attrgetter('attribute', 'value')))
    identity_assignment_proto.allowIdentitySelfSelection = True
    return [identity_assignment_proto]