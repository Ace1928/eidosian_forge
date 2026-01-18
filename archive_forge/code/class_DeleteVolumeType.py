import functools
import logging
from cinderclient import api_versions
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class DeleteVolumeType(command.Command):
    _description = _('Delete volume type(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteVolumeType, self).get_parser(prog_name)
        parser.add_argument('volume_types', metavar='<volume-type>', nargs='+', help=_('Volume type(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        result = 0
        for volume_type in parsed_args.volume_types:
            try:
                vol_type = utils.find_resource(volume_client.volume_types, volume_type)
                volume_client.volume_types.delete(vol_type)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete volume type with name or ID '%(volume_type)s': %(e)s") % {'volume_type': volume_type, 'e': e})
        if result > 0:
            total = len(parsed_args.volume_types)
            msg = _('%(result)s of %(total)s volume types failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)