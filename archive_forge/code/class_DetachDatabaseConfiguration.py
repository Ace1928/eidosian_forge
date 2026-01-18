import json
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
class DetachDatabaseConfiguration(command.Command):
    _description = _('Detaches a configuration group from an instance.')

    def get_parser(self, prog_name):
        parser = super(DetachDatabaseConfiguration, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
        return parser

    def take_action(self, parsed_args):
        db_instances = self.app.client_manager.database.instances
        instance_id = parsed_args.instance
        if not uuidutils.is_uuid_like(instance_id):
            instance_id = osc_utils.find_resource(db_instances, instance_id)
        db_instances.update(instance_id, remove_configuration=True)