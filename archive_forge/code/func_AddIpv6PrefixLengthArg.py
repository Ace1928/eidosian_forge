from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddIpv6PrefixLengthArg(parser):
    parser.add_argument('--ipv6-prefix-length', type=int, help='\n        The prefix length of the external IPv6 address range. This flag should be used together\n        with `--ipv6-address`. Currently only `/96` is supported and the default value\n        is `96`.\n      ')