import argparse
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
class ResizeDatabaseInstanceVolume(command.Command):
    _description = _('Resizes the volume size of an instance.')

    def get_parser(self, prog_name):
        parser = super(ResizeDatabaseInstanceVolume, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
        parser.add_argument('size', metavar='<size>', type=int, default=None, help=_('New size of the instance disk volume in GB.'))
        return parser

    def take_action(self, parsed_args):
        db_instances = self.app.client_manager.database.instances
        instance = osc_utils.find_resource(db_instances, parsed_args.instance)
        db_instances.resize_volume(instance, parsed_args.size)