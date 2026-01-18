import argparse
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
class PromoteDatabaseInstanceToReplicaSource(command.Command):
    _description = _('Promotes a replica to be the new replica source of its set.')

    def get_parser(self, prog_name):
        parser = super(PromoteDatabaseInstanceToReplicaSource, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
        return parser

    def take_action(self, parsed_args):
        db_instances = self.app.client_manager.database.instances
        instance = osc_utils.find_resource(db_instances, parsed_args.instance)
        db_instances.promote_to_replica_source(instance)