import logging
from openstackclient.identity import common as identity_common
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
class ShowResourceLock(command.ShowOne):
    """Show details about a resource lock."""
    _description = _('Show details about a resource lock')

    def get_parser(self, prog_name):
        parser = super(ShowResourceLock, self).get_parser(prog_name)
        parser.add_argument('lock', metavar='<lock>', help=_('ID of resource lock to show.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        resource_lock = apiutils.find_resource(share_client.resource_locks, parsed_args.lock)
        return (LOCK_DETAIL_ATTRIBUTES, osc_utils.get_dict_properties(resource_lock._info, LOCK_DETAIL_ATTRIBUTES))