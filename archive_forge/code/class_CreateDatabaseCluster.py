from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from troveclient.i18n import _
from troveclient.v1.shell import _parse_extended_properties
from troveclient.v1.shell import _parse_instance_options
from troveclient.v1.shell import EXT_PROPS_HELP
from troveclient.v1.shell import EXT_PROPS_METAVAR
from troveclient.v1.shell import INSTANCE_HELP
from troveclient.v1.shell import INSTANCE_METAVAR
class CreateDatabaseCluster(command.ShowOne):
    _description = _('Creates a new database cluster.')

    def get_parser(self, prog_name):
        parser = super(CreateDatabaseCluster, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', type=str, help=_('Name of the cluster.'))
        parser.add_argument('datastore', metavar='<datastore>', help=_('A datastore name or ID.'))
        parser.add_argument('datastore_version', metavar='<datastore_version>', help=_('A datastore version name or ID.'))
        parser.add_argument('--instance', metavar=INSTANCE_METAVAR, action='append', dest='instances', default=[], help=INSTANCE_HELP)
        parser.add_argument('--locality', metavar='<policy>', default=None, choices=['affinity', 'anti-affinity'], help=_('Locality policy to use when creating cluster. Choose one of %(choices)s.'))
        parser.add_argument('--extended-properties', dest='extended_properties', metavar=EXT_PROPS_METAVAR, default=None, help=EXT_PROPS_HELP)
        parser.add_argument('--configuration', metavar='<configuration>', type=str, default=None, help=_('ID of the configuration group to attach to the cluster.'))
        return parser

    def take_action(self, parsed_args):
        database = self.app.client_manager.database
        instances = _parse_instance_options(database, parsed_args.instances)
        extended_properties = {}
        if parsed_args.extended_properties:
            extended_properties = _parse_extended_properties(parsed_args.extended_properties)
        cluster = database.clusters.create(parsed_args.name, parsed_args.datastore, parsed_args.datastore_version, instances=instances, locality=parsed_args.locality, extended_properties=extended_properties, configuration=parsed_args.configuration)
        cluster = set_attributes_for_print_detail(cluster)
        return zip(*sorted(cluster.items()))