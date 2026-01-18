from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddSrcSecureTags(parser, required=False):
    """Adds a  source secure tag to this rule."""
    parser.add_argument('--src-secure-tags', type=arg_parsers.ArgList(), metavar='SOURCE_SECURE_TAGS', required=required, help='A list of instance secure tags indicating the set of instances on the network to which the rule applies if all other fields match. Either --src-ip-ranges or --src-secure-tags must be specified for ingress traffic. If both --src-ip-ranges and --src-secure-tags are specified, an inbound connection is allowed if either the range of the source matches --src-ip-ranges or the tag of the source matches --src-secure-tags. Secure Tags can be assigned to instances during instance creation.')