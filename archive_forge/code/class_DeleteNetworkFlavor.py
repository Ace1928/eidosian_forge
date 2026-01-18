import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class DeleteNetworkFlavor(command.Command):
    _description = _('Delete network flavors')

    def get_parser(self, prog_name):
        parser = super(DeleteNetworkFlavor, self).get_parser(prog_name)
        parser.add_argument('flavor', metavar='<flavor>', nargs='+', help=_('Flavor(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for flavor in parsed_args.flavor:
            try:
                obj = client.find_flavor(flavor, ignore_missing=False)
                client.delete_flavor(obj)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete flavor with name or ID '%(flavor)s': %(e)s"), {'flavor': flavor, 'e': e})
        if result > 0:
            total = len(parsed_args.flavor)
            msg = _('%(result)s of %(total)s flavors failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)