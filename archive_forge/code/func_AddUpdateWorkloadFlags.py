from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.base import ReleaseTrack
from googlecloudsdk.command_lib.assured import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddUpdateWorkloadFlags(parser):
    """Method to add update workload flags."""
    AddWorkloadResourceArgToParser(parser, verb='update')
    parser.add_argument('--etag', help='The etag acquired by reading the Assured Workloads environment before updating.')
    updatable_fields = parser.add_group(required=True, help='Settings that can be updated on the Assured Workloads environment.')
    updatable_fields.add_argument('--display-name', help='The new display name of the Assured Workloads environment.')
    updatable_fields.add_argument('--violation-notifications-enabled', help='The notification setting of the Assured Workloads environment.')
    updatable_fields.add_argument('--labels', metavar='KEY=VALUE', type=arg_parsers.ArgDict(), help='The new labels of the Assured Workloads environment, for example, LabelKey1=LabelValue1,LabelKey2=LabelValue2')