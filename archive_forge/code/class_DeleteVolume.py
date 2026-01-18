import argparse
import copy
import functools
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class DeleteVolume(command.Command):
    _description = _('Delete volume(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteVolume, self).get_parser(prog_name)
        parser.add_argument('volumes', metavar='<volume>', nargs='+', help=_('Volume(s) to delete (name or ID)'))
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--force', action='store_true', help=_('Attempt forced removal of volume(s), regardless of state (defaults to False)'))
        group.add_argument('--purge', action='store_true', help=_('Remove any snapshots along with volume(s) (defaults to False)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        result = 0
        for i in parsed_args.volumes:
            try:
                volume_obj = utils.find_resource(volume_client.volumes, i)
                if parsed_args.force:
                    volume_client.volumes.force_delete(volume_obj.id)
                else:
                    volume_client.volumes.delete(volume_obj.id, cascade=parsed_args.purge)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete volume with name or ID '%(volume)s': %(e)s"), {'volume': i, 'e': e})
        if result > 0:
            total = len(parsed_args.volumes)
            msg = _('%(result)s of %(total)s volumes failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)