import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
from manilaclient.osc import utils
class UnsetShareType(command.Command):
    """Unset share type extra specs."""
    _description = _('Unset share type extra specs')

    def get_parser(self, prog_name):
        parser = super(UnsetShareType, self).get_parser(prog_name)
        parser.add_argument('share_type', metavar='<share_type>', help=_('Name or ID of the share type to modify'))
        parser.add_argument('extra_specs', metavar='<key>', nargs='+', help=_('Remove extra_specs from this share type'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_type = apiutils.find_resource(share_client.share_types, parsed_args.share_type)
        if parsed_args.extra_specs:
            try:
                share_type.unset_keys(parsed_args.extra_specs)
            except Exception as e:
                raise exceptions.CommandError('Failed to remove share type extra spec: %s' % e)