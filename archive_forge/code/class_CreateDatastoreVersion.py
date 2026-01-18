from osc_lib.command import command
from osc_lib import utils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient import utils as tc_utils
class CreateDatastoreVersion(command.Command):
    _description = _('Creates a datastore version.')

    def get_parser(self, prog_name):
        parser = super(CreateDatastoreVersion, self).get_parser(prog_name)
        parser.add_argument('version_name', help=_('Datastore version name.'))
        parser.add_argument('datastore_name', help=_('Datastore name. The datastore is created automatically if does not exist.'))
        parser.add_argument('datastore_manager', help=_('Datastore manager, e.g. mysql'))
        parser.add_argument('image_id', help=_('ID of the datastore image in Glance. This can be empty string if --image-tags is specified.'))
        parser.add_argument('--active', action='store_true', help=_('Enable the datastore version.'))
        parser.add_argument('--image-tags', help=_('List of image tags separated by comma, e.g. trove,mysql'))
        parser.add_argument('--default', action='store_true', help=_('If set the datastore version as default.'))
        parser.add_argument('--version-number', help=_('The version number for the database. If not specified, use the version name as the default value.'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.database.mgmt_ds_versions
        image_tags = []
        if parsed_args.image_tags:
            image_tags = parsed_args.image_tags.split(',')
        try:
            client.create(parsed_args.version_name, parsed_args.datastore_name, parsed_args.datastore_manager, parsed_args.image_id, image_tags=image_tags, active='true' if parsed_args.active else 'false', default='true' if parsed_args.default else 'false', version=parsed_args.version_number)
        except Exception as e:
            msg = _('Failed to create datastore version %(version)s: %(e)s') % {'version': parsed_args.version_name, 'e': e}
            raise exceptions.CommandError(msg)