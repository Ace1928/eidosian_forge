import argparse
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
class DeleteDatabaseInstance(base.TroveDeleter):
    _description = _('Deletes an instance.')

    def get_parser(self, prog_name):
        parser = super(DeleteDatabaseInstance, self).get_parser(prog_name)
        parser.add_argument('instance', nargs='+', metavar='instance', help='Id or name of instance(s).')
        parser.add_argument('--force', action='store_true', default=False, help=_('Force delete the instance, will reset the instance status before deleting.'))
        return parser

    def take_action(self, parsed_args):
        db_instances = self.app.client_manager.database.instances
        self.delete_func = db_instances.force_delete if parsed_args.force else db_instances.delete
        self.resource = 'database instance'
        ids = []
        for instance_id in parsed_args.instance:
            if not uuidutils.is_uuid_like(instance_id):
                try:
                    instance_id = trove_utils.get_resource_id_by_name(db_instances, instance_id)
                except Exception as e:
                    msg = 'Failed to get database instance %s, error: %s' % (instance_id, str(e))
                    raise exceptions.CommandError(msg)
            ids.append(instance_id)
        self.delete_resources(ids)