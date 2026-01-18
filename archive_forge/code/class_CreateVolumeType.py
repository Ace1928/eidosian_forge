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
class CreateVolumeType(command.ShowOne):
    _description = _('Create new volume type')

    def get_parser(self, prog_name):
        parser = super(CreateVolumeType, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Volume type name'))
        parser.add_argument('--description', metavar='<description>', help=_('Volume type description'))
        public_group = parser.add_mutually_exclusive_group()
        public_group.add_argument('--public', action='store_true', dest='is_public', default=None, help=_('Volume type is accessible to the public'))
        public_group.add_argument('--private', action='store_false', dest='is_public', default=None, help=_('Volume type is not accessible to the public'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, dest='properties', help=_('Set a property on this volume type (repeat option to set multiple properties)'))
        parser.add_argument('--multiattach', action='store_true', default=False, help=_("Enable multi-attach for this volume type (this is an alias for '--property multiattach=<is> True') (requires driver support)"))
        parser.add_argument('--cacheable', action='store_true', default=False, help=_("Enable caching for this volume type (this is an alias for '--property cacheable=<is> True') (requires driver support)"))
        parser.add_argument('--replicated', action='store_true', default=False, help=_("Enabled replication for this volume type (this is an alias for '--property replication_enabled=<is> True') (requires driver support)"))
        parser.add_argument('--availability-zone', action='append', dest='availability_zones', help=_("Set an availability zone for this volume type (this is an alias for '--property RESKEY:availability_zones:<az>') (repeat option to set multiple availability zones)"))
        parser.add_argument('--project', metavar='<project>', help=_('Allow <project> to access private type (name or ID) (must be used with --private option)'))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--encryption-provider', metavar='<provider>', help=_('Set the encryption provider format for this volume type (e.g "luks" or "plain") (admin only) (this option is required when setting encryption type of a volume; consider using other encryption options such as: "--encryption-cipher", "--encryption-key-size" and "--encryption-control-location")'))
        parser.add_argument('--encryption-cipher', metavar='<cipher>', help=_('Set the encryption algorithm or mode for this volume type (e.g "aes-xts-plain64") (admin only)'))
        parser.add_argument('--encryption-key-size', metavar='<key-size>', type=int, help=_('Set the size of the encryption key of this volume type (e.g "128" or "256") (admin only)'))
        parser.add_argument('--encryption-control-location', metavar='<control-location>', choices=['front-end', 'back-end'], help=_('Set the notional service where the encryption is performed ("front-end" or "back-end") (admin only) (The default value for this option is "front-end" when setting encryption type of a volume. Consider using other encryption options such as: "--encryption-cipher", "--encryption-key-size" and "--encryption-provider")'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        volume_client = self.app.client_manager.volume
        if parsed_args.project and parsed_args.is_public is not False:
            msg = _('--project is only allowed with --private')
            raise exceptions.CommandError(msg)
        kwargs = {}
        if parsed_args.is_public is not None:
            kwargs['is_public'] = parsed_args.is_public
        volume_type = volume_client.volume_types.create(parsed_args.name, description=parsed_args.description, **kwargs)
        volume_type._info.pop('extra_specs')
        if parsed_args.project:
            try:
                project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
                volume_client.volume_type_access.add_project_access(volume_type.id, project_id)
            except Exception as e:
                msg = _('Failed to add project %(project)s access to type: %(e)s')
                LOG.error(msg % {'project': parsed_args.project, 'e': e})
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
            result = volume_type.set_keys(properties)
            volume_type._info.update({'properties': format_columns.DictColumn(result)})
        if parsed_args.encryption_provider or parsed_args.encryption_cipher or parsed_args.encryption_key_size or parsed_args.encryption_control_location:
            try:
                encryption = _create_encryption_type(volume_client, volume_type, parsed_args)
            except Exception as e:
                LOG.error(_('Failed to set encryption information for this volume type: %s'), e)
            encryption._info.pop('volume_type_id', None)
            volume_type._info.update({'encryption': format_columns.DictColumn(encryption._info)})
        volume_type._info.pop('os-volume-type-access:is_public', None)
        return zip(*sorted(volume_type._info.items()))