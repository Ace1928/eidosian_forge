import argparse
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
class RebootDatabaseInstance(command.Command):
    _description = _('Reboots an instance(the Nova server).')

    def get_parser(self, prog_name):
        parser = super(RebootDatabaseInstance, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
        return parser

    def take_action(self, parsed_args):
        instance_id = parsed_args.instance
        if not uuidutils.is_uuid_like(instance_id):
            instance_mgr = self.app.client_manager.database.instances
            instance_id = osc_utils.find_resource(instance_mgr, instance_id)
        mgmt_instance_mgr = self.app.client_manager.database.mgmt_instances
        mgmt_instance_mgr.reboot(instance_id)