import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import exceptions as apiclient_exceptions
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
class UnsetShare(command.Command):
    """Unset share properties."""
    _description = _('Unset share properties')

    def get_parser(self, prog_name):
        parser = super(UnsetShare, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Share to modify (name or ID)'))
        parser.add_argument('--property', metavar='<key>', action='append', help=_('Remove a property from share (repeat option to remove multiple properties)'))
        parser.add_argument('--name', action='store_true', help=_('Unset share name.'))
        parser.add_argument('--description', action='store_true', help=_('Unset share description.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_obj = apiutils.find_resource(share_client.shares, parsed_args.share)
        result = 0
        kwargs = {}
        if parsed_args.name:
            kwargs['display_name'] = None
        if parsed_args.description:
            kwargs['display_description'] = None
        if kwargs:
            try:
                share_client.shares.update(share_obj.id, **kwargs)
            except Exception as e:
                LOG.error(_('Failed to unset share display name or display description: %s'), e)
                result += 1
        if parsed_args.property:
            for key in parsed_args.property:
                try:
                    share_obj.delete_metadata([key])
                except Exception as e:
                    LOG.error(_("Failed to unset share property '%(key)s': %(e)s"), {'key': key, 'e': e})
                    result += 1
        if result > 0:
            raise exceptions.CommandError(_('One or more of the unset operations failed'))