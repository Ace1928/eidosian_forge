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
def _add_default_options(parser):
    default_group = parser.add_mutually_exclusive_group()
    default_group.add_argument('--default', action='store_true', help=_('Set this as a default subnet pool'))
    default_group.add_argument('--no-default', action='store_true', help=_('Set this as a non-default subnet pool'))