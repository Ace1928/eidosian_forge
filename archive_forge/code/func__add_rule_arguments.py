import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
def _add_rule_arguments(parser):
    parser.add_argument('--max-kbps', dest='max_kbps', metavar='<max-kbps>', type=int, help=_('Maximum bandwidth in kbps'))
    parser.add_argument('--max-burst-kbits', dest='max_burst_kbits', metavar='<max-burst-kbits>', type=int, help=_('Maximum burst in kilobits, 0 or not specified means automatic, which is 80%% of the bandwidth limit, which works for typical TCP traffic. For details check the QoS user workflow.'))
    parser.add_argument('--dscp-mark', dest='dscp_mark', metavar='<dscp-mark>', type=int, help=_('DSCP mark: value can be 0, even numbers from 8-56, excluding 42, 44, 50, 52, and 54'))
    parser.add_argument('--min-kbps', dest='min_kbps', metavar='<min-kbps>', type=int, help=_('Minimum guaranteed bandwidth in kbps'))
    parser.add_argument('--min-kpps', dest='min_kpps', metavar='<min-kpps>', type=int, help=_('Minimum guaranteed packet rate in kpps'))
    direction_group = parser.add_mutually_exclusive_group()
    direction_group.add_argument('--ingress', action='store_true', help=_('Ingress traffic direction from the project point of view'))
    direction_group.add_argument('--egress', action='store_true', help=_('Egress traffic direction from the project point of view'))
    direction_group.add_argument('--any', action='store_true', help=_('Any traffic direction from the project point of view. Can be used only with minimum packet rate rule.'))