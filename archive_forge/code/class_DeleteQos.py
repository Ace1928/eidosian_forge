import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteQos(command.Command):
    _description = _('Delete QoS specification')

    def get_parser(self, prog_name):
        parser = super(DeleteQos, self).get_parser(prog_name)
        parser.add_argument('qos_specs', metavar='<qos-spec>', nargs='+', help=_('QoS specification(s) to delete (name or ID)'))
        parser.add_argument('--force', action='store_true', default=False, help=_('Allow to delete in-use QoS specification(s)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        result = 0
        for i in parsed_args.qos_specs:
            try:
                qos_spec = utils.find_resource(volume_client.qos_specs, i)
                volume_client.qos_specs.delete(qos_spec.id, parsed_args.force)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete QoS specification with name or ID '%(qos)s': %(e)s") % {'qos': i, 'e': e})
        if result > 0:
            total = len(parsed_args.qos_specs)
            msg = _('%(result)s of %(total)s QoS specifications failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)