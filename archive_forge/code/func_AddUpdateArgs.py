from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from googlecloudsdk.api_lib.edge_cloud.networking import utils
from googlecloudsdk.calliope import arg_parsers
def AddUpdateArgs(parser):
    """Adds arguments for Update."""

    def helptext(verb, prep):
        return '{} the comma-separated list of CIDRs {} the set of range advertisements.'.format(verb, prep)

    def cidrlist(argstr):
        split = argstr.split(',')
        parsed = map(ipaddress.ip_network, split)
        retlist = sorted(parsed)
        retset = set(retlist)
        if len(retlist) != len(retset):
            raise ValueError('CIDR list contained duplicates.')
        return retlist
    adv_group = parser.add_argument_group(mutex=True)
    adv_group.add_argument('--add-advertisement-ranges', help=helptext('add', 'to'), type=cidrlist, default=[])
    adv_group.add_argument('--set-advertisement-ranges', help=helptext('replace', 'with'), type=cidrlist, default=[])
    adv_group.add_argument('--remove-advertisement-ranges', help=helptext('remove', 'from'), type=cidrlist, default=[])