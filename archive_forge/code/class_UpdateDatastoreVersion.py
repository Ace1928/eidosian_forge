from osc_lib.command import command
from osc_lib import utils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient import utils as tc_utils
class UpdateDatastoreVersion(command.Command):
    _description = _('Updates a datastore version.')

    def get_parser(self, prog_name):
        parser = super(UpdateDatastoreVersion, self).get_parser(prog_name)
        parser.add_argument('datastore_version_id', help=_('Datastore version ID.'))
        parser.add_argument('--datastore-manager', default=None, help=_('Datastore manager name.'))
        parser.add_argument('--image', default=None, help=_('ID of the datastore image in Glance.'))
        parser.add_argument('--image-tags', default=None, help=_('List of image tags separated by comma, e.g. trove,mysql'))
        parser.add_argument('--version-name', help=_('New datastore version name.'))
        enable_group = parser.add_mutually_exclusive_group()
        enable_group.add_argument('--enable', dest='enable', default=None, action='store_const', const='true')
        enable_group.add_argument('--disable', dest='enable', default=None, action='store_const', const='false')
        default_group = parser.add_mutually_exclusive_group()
        default_group.add_argument('--default', dest='default', default=None, action='store_const', const='true')
        default_group.add_argument('--non-default', dest='default', default=None, action='store_const', const='false')
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.database.mgmt_ds_versions
        image_tags = None
        if parsed_args.image_tags is not None:
            image_tags = parsed_args.image_tags.split(',')
        try:
            client.edit(parsed_args.datastore_version_id, datastore_manager=parsed_args.datastore_manager, image=parsed_args.image, image_tags=image_tags, active=parsed_args.enable, default=parsed_args.default, name=parsed_args.version_name)
        except Exception as e:
            msg = _('Failed to update datastore version %(version)s: %(e)s') % {'version': parsed_args.datastore_version_id, 'e': e}
            raise exceptions.CommandError(msg)