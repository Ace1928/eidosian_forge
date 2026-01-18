import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class DeleteMeter(command.Command):
    _description = _('Delete network meter')

    def get_parser(self, prog_name):
        parser = super(DeleteMeter, self).get_parser(prog_name)
        parser.add_argument('meter', metavar='<meter>', nargs='+', help=_('Meter to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for meter in parsed_args.meter:
            try:
                obj = client.find_metering_label(meter, ignore_missing=False)
                client.delete_metering_label(obj)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete meter with ID '%(meter)s': %(e)s"), {'meter': meter, 'e': e})
        if result > 0:
            total = len(parsed_args.meter)
            msg = _('%(result)s of %(total)s meters failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)