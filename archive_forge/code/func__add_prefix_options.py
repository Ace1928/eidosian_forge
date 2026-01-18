import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _add_prefix_options(parser, for_create=False):
    parser.add_argument('--pool-prefix', metavar='<pool-prefix>', dest='prefixes', action='append', required=for_create, help=_('Set subnet pool prefixes (in CIDR notation) (repeat option to set multiple prefixes)'))
    parser.add_argument('--default-prefix-length', metavar='<default-prefix-length>', type=int, action=parseractions.NonNegativeAction, help=_('Set subnet pool default prefix length'))
    parser.add_argument('--min-prefix-length', metavar='<min-prefix-length>', action=parseractions.NonNegativeAction, type=int, help=_('Set subnet pool minimum prefix length'))
    parser.add_argument('--max-prefix-length', metavar='<max-prefix-length>', type=int, action=parseractions.NonNegativeAction, help=_('Set subnet pool maximum prefix length'))