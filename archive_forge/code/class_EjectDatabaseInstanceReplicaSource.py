import argparse
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
class EjectDatabaseInstanceReplicaSource(command.Command):
    _description = _('Ejects a replica source from its set.')

    def get_parser(self, prog_name):
        parser = super(EjectDatabaseInstanceReplicaSource, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
        return parser

    def take_action(self, parsed_args):
        db_instances = self.app.client_manager.database.instances
        instance = osc_utils.find_resource(db_instances, parsed_args.instance)
        db_instances.eject_replica_source(instance)