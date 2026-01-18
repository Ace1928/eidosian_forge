import logging
import netaddr
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class DeleteAddressGroup(command.Command):
    _description = _('Delete address group(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteAddressGroup, self).get_parser(prog_name)
        parser.add_argument('address_group', metavar='<address-group>', nargs='+', help=_('Address group(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for group in parsed_args.address_group:
            try:
                obj = client.find_address_group(group, ignore_missing=False)
                client.delete_address_group(obj)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete address group with name or ID '%(group)s': %(e)s"), {'group': group, 'e': e})
        if result > 0:
            total = len(parsed_args.address_group)
            msg = _('%(result)s of %(total)s address groups failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)