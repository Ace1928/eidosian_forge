from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
class TrafficFlags(BinaryCommandFlag):
    """Encapsulates flags to configure traffic routes to the service."""

    def __init__(self):
        super(TrafficFlags, self).__init__()
        self.to_revisions_flag = StringListFlag('--to-revisions', metavar='REVISION-NAME=PERCENTAGE', help='Comma-separated list of traffic assignments in the form REVISION-NAME=PERCENTAGE. REVISION-NAME must be the name for a revision for the service as returned by \'gcloud kuberun core revisions list\'. PERCENTAGE must be an integer percentage between 0 and 100 inclusive. E.g. service-nw9hs=10,service-nw9hs=20 Up to 100 percent of traffic may be assigned. If 100 percent of traffic is assigned, the Service traffic is updated as specified. If under 100 percent of traffic is assigned, the Service traffic is updated as specified for revisions with assignments and traffic is scaled up or down down proportionally as needed for revision that are currently serving traffic but that do not have new assignments. For example assume revision-1 is serving 40 percent of traffic and revision-2 is serving 60 percent. If revision-1 is assigned 45 percent of traffic and no assignment is made for revision-2, the service is updated with revsion-1 assigned 45 percent of traffic and revision-2 scaled down to 55 percent. You can use "LATEST" as a special revision name to always put the given percentage of traffic on the latest ready revision.')

    def AddToParser(self, parser):
        mutex_group = parser.add_mutually_exclusive_group(required=True)
        mutex_group.add_argument('--to-latest', default=False, action='store_true', help="If true, assign 100 percent of traffic to the 'latest' revision of this service. Note that when a new revision is created, it will become the 'latest' and traffic will be directed to it. Defaults to False. Synonymous with `--to-revisions=LATEST=100`.")
        self.to_revisions_flag.AddToParser(mutex_group)

    def FormatFlags(self, args):
        if args.IsSpecified('to_latest'):
            return ['--to-latest']
        elif args.IsSpecified('to_revisions'):
            return self.to_revisions_flag.FormatFlags(args)