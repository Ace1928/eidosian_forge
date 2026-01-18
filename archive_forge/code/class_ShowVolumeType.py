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
class ShowVolumeType(command.ShowOne):
    _description = _('Display volume type details')

    def get_parser(self, prog_name):
        parser = super(ShowVolumeType, self).get_parser(prog_name)
        parser.add_argument('volume_type', metavar='<volume-type>', help=_('Volume type to display (name or ID)'))
        parser.add_argument('--encryption-type', action='store_true', help=_('Display encryption information of this volume type (admin only)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        volume_type = utils.find_resource(volume_client.volume_types, parsed_args.volume_type)
        properties = format_columns.DictColumn(volume_type._info.pop('extra_specs', {}))
        volume_type._info.update({'properties': properties})
        access_project_ids = None
        if not volume_type.is_public:
            try:
                volume_type_access = volume_client.volume_type_access.list(volume_type.id)
                project_ids = [utils.get_field(item, 'project_id') for item in volume_type_access]
                access_project_ids = format_columns.ListColumn(project_ids)
            except Exception as e:
                msg = _('Failed to get access project list for volume type %(type)s: %(e)s')
                LOG.error(msg % {'type': volume_type.id, 'e': e})
        volume_type._info.update({'access_project_ids': access_project_ids})
        if parsed_args.encryption_type:
            try:
                encryption = volume_client.volume_encryption_types.get(volume_type.id)
                encryption._info.pop('volume_type_id', None)
                volume_type._info.update({'encryption': format_columns.DictColumn(encryption._info)})
            except Exception as e:
                LOG.error(_('Failed to display the encryption information of this volume type: %s'), e)
        volume_type._info.pop('os-volume-type-access:is_public', None)
        return zip(*sorted(volume_type._info.items()))