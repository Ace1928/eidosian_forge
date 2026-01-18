from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddTagsArgs(parser):
    parser.add_argument('--tags', type=arg_parsers.ArgList(min_length=1), metavar='TAG', help="      Specifies a list of tags to apply to the VMs created from the\n      imported machine image. These tags allow network firewall rules and routes\n      to be applied to specified VMs. See\n      gcloud_compute_firewall-rules_create(1) for more details.\n\n      To read more about configuring network tags, read this guide:\n      https://cloud.google.com/vpc/docs/add-remove-network-tags\n\n      To list VMs with their respective status and tags, run:\n\n        $ gcloud compute instances list --format='table(name,status,tags.list())'\n\n      To list VMs tagged with a specific tag, `tag1`, run:\n\n        $ gcloud compute instances list --filter='tags:tag1'\n      ")