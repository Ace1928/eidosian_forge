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
class ListVolumeType(command.Lister):
    _description = _('List volume types')

    def get_parser(self, prog_name):
        parser = super(ListVolumeType, self).get_parser(prog_name)
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        public_group = parser.add_mutually_exclusive_group()
        public_group.add_argument('--default', action='store_true', default=False, help=_('List the default volume type'))
        public_group.add_argument('--public', action='store_true', dest='is_public', default=None, help=_('List only public types'))
        public_group.add_argument('--private', action='store_false', dest='is_public', default=None, help=_('List only private types (admin only)'))
        parser.add_argument('--encryption-type', action='store_true', help=_('Display encryption information for each volume type (admin only)'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, dest='properties', help=_('Filter by a property on the volume types (repeat option to filter by multiple properties) (admin only except for user-visible extra specs) (supported by --os-volume-api-version 3.52 or above)'))
        parser.add_argument('--multiattach', action='store_true', default=False, help=_("List only volume types with multi-attach enabled (this is an alias for '--property multiattach=<is> True') (supported by --os-volume-api-version 3.52 or above)"))
        parser.add_argument('--cacheable', action='store_true', default=False, help=_("List only volume types with caching enabled (this is an alias for '--property cacheable=<is> True') (admin only) (supported by --os-volume-api-version 3.52 or above)"))
        parser.add_argument('--replicated', action='store_true', default=False, help=_("List only volume types with replication enabled (this is an alias for '--property replication_enabled=<is> True') (supported by --os-volume-api-version 3.52 or above)"))
        parser.add_argument('--availability-zone', action='append', dest='availability_zones', help=_("List only volume types with this availability configured (this is an alias for '--property RESKEY:availability_zones:<az>') (repeat option to filter on multiple availability zones)"))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        if parsed_args.long:
            columns = ['ID', 'Name', 'Is Public', 'Description', 'Extra Specs']
            column_headers = ['ID', 'Name', 'Is Public', 'Description', 'Properties']
        else:
            columns = ['ID', 'Name', 'Is Public']
            column_headers = ['ID', 'Name', 'Is Public']
        if parsed_args.default:
            data = [volume_client.volume_types.default()]
        else:
            search_opts = {}
            properties = {}
            if parsed_args.properties:
                properties.update(parsed_args.properties)
            if parsed_args.multiattach:
                properties['multiattach'] = '<is> True'
            if parsed_args.cacheable:
                properties['cacheable'] = '<is> True'
            if parsed_args.replicated:
                properties['replication_enabled'] = '<is> True'
            if parsed_args.availability_zones:
                properties['RESKEY:availability_zones'] = ','.join(parsed_args.availability_zones)
            if properties:
                if volume_client.api_version < api_versions.APIVersion('3.52'):
                    msg = _("--os-volume-api-version 3.52 or greater is required to use the '--property' option or any of the alias options")
                    raise exceptions.CommandError(msg)
                search_opts['extra_specs'] = properties
            data = volume_client.volume_types.list(search_opts=search_opts, is_public=parsed_args.is_public)
        formatters = {'Extra Specs': format_columns.DictColumn}
        if parsed_args.encryption_type:
            encryption = {}
            for d in volume_client.volume_encryption_types.list():
                volume_type_id = d._info['volume_type_id']
                del_key = ['deleted', 'created_at', 'updated_at', 'deleted_at', 'volume_type_id']
                for key in del_key:
                    d._info.pop(key, None)
                encryption[volume_type_id] = d._info
            columns += ['id']
            column_headers += ['Encryption']
            _EncryptionInfoColumn = functools.partial(EncryptionInfoColumn, encryption_data=encryption)
            formatters['id'] = _EncryptionInfoColumn
        return (column_headers, (utils.get_item_properties(s, columns, formatters=formatters) for s in data))