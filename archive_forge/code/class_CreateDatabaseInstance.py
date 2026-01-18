import argparse
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
class CreateDatabaseInstance(command.ShowOne):
    _description = _('Creates a new database instance.')

    def get_parser(self, prog_name):
        parser = super(CreateDatabaseInstance, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name of the instance.'))
        parser.add_argument('--flavor', metavar='<flavor>', type=str, help=_('Flavor to create the instance (name or ID). Flavor is not required when creating replica instances.'))
        parser.add_argument('--size', metavar='<size>', type=int, default=None, help=_('Size of the instance disk volume in GB. Required when volume support is enabled.'))
        parser.add_argument('--volume-type', metavar='<volume_type>', type=str, default=None, help=_('Volume type. Optional when volume support is enabled.'))
        parser.add_argument('--databases', metavar='<database>', nargs='+', default=[], help=_('Optional list of databases.'))
        parser.add_argument('--users', metavar='<user:password>', nargs='+', default=[], help=_('Optional list of users.'))
        parser.add_argument('--backup', metavar='<backup>', default=None, help=_('A backup name or ID.'))
        parser.add_argument('--availability-zone', metavar='<availability_zone>', default=None, help=_('The Zone hint to give to Nova.'))
        parser.add_argument('--datastore', metavar='<datastore>', default=None, help=_('A datastore name or ID.'))
        parser.add_argument('--datastore-version', metavar='<datastore_version>', default=None, help=_('A datastore version name or ID.'))
        parser.add_argument('--datastore-version-number', default=None, help=_('The version number for the database. The version number is needed for the datastore versions with the same name.'))
        parser.add_argument('--nic', metavar='<net-id=<net-uuid>,subnet-id=<subnet-uuid>,ip-address=<ip-address>>', dest='nics', help=_('Create instance in the given Neutron network. This information is used for creating user-facing port for the instance. Either network ID or subnet ID (or both) should be specified, IP address is optional'))
        parser.add_argument('--configuration', metavar='<configuration>', default=None, help=_('ID of the configuration group to attach to the instance.'))
        parser.add_argument('--replica-of', metavar='<source_instance>', default=None, help=_('ID or name of an existing instance to replicate from.'))
        parser.add_argument('--replica-count', metavar='<count>', type=int, default=None, help=_('Number of replicas to create (defaults to 1 if replica_of specified).'))
        parser.add_argument('--module', metavar='<module>', type=str, dest='modules', action='append', default=[], help=_('ID or name of the module to apply.  Specify multiple times to apply multiple modules.'))
        parser.add_argument('--locality', metavar='<policy>', default=None, choices=['affinity', 'anti-affinity'], help=_('Locality policy to use when creating replicas. Choose one of %(choices)s.'))
        parser.add_argument('--region', metavar='<region>', type=str, default=None, help=argparse.SUPPRESS)
        parser.add_argument('--is-public', action='store_true', help='Whether or not to make the instance public.')
        parser.add_argument('--allowed-cidr', action='append', dest='allowed_cidrs', help='The IP CIDRs that are allowed to access the database instance. Repeat for multiple values')
        return parser

    def take_action(self, parsed_args):
        database = self.app.client_manager.database
        db_instances = database.instances
        if not parsed_args.replica_of and (not parsed_args.flavor):
            raise exceptions.CommandError(_('Please specify a flavor'))
        if parsed_args.replica_of and parsed_args.flavor:
            print('Warning: Flavor is ignored for creating replica.')
        if not parsed_args.replica_of:
            flavor_id = osc_utils.find_resource(database.flavors, parsed_args.flavor).id
        else:
            flavor_id = None
        volume = None
        if parsed_args.size is not None and parsed_args.size <= 0:
            raise exceptions.ValidationError(_("Volume size '%s' must be an integer and greater than 0.") % parsed_args.size)
        elif parsed_args.size:
            volume = {'size': parsed_args.size, 'type': parsed_args.volume_type}
        restore_point = None
        if parsed_args.backup:
            restore_point = {'backupRef': osc_utils.find_resource(database.backups, parsed_args.backup).id}
        replica_of = None
        replica_count = parsed_args.replica_count
        if parsed_args.replica_of:
            replica_of = osc_utils.find_resource(db_instances, parsed_args.replica_of)
            replica_count = replica_count or 1
        locality = None
        if parsed_args.locality:
            locality = parsed_args.locality
            if replica_of:
                raise exceptions.ValidationError(_('Cannot specify locality when adding replicas to existing master.'))
        databases = [{'name': value} for value in parsed_args.databases]
        users = [{'name': n, 'password': p, 'databases': databases} for n, p in [z.split(':')[:2] for z in parsed_args.users]]
        nics = []
        if parsed_args.nics:
            nic_info = {}
            allowed_keys = {'net-id': 'network_id', 'subnet-id': 'subnet_id', 'ip-address': 'ip_address'}
            fields = parsed_args.nics.split(',')
            for field in fields:
                field = field.strip()
                k, v = field.split('=', 1)
                k = k.strip()
                v = v.strip()
                if k not in allowed_keys.keys():
                    raise exceptions.ValidationError(f'{k} is not allowed.')
                if v:
                    nic_info[allowed_keys[k]] = v
            nics.append(nic_info)
        modules = []
        for module in parsed_args.modules:
            modules.append(osc_utils.find_resource(database.modules, module).id)
        access = {'is_public': False}
        if parsed_args.is_public:
            access['is_public'] = True
        if parsed_args.allowed_cidrs:
            access['allowed_cidrs'] = parsed_args.allowed_cidrs
        instance = db_instances.create(parsed_args.name, flavor_id=flavor_id, volume=volume, databases=databases, users=users, restorePoint=restore_point, availability_zone=parsed_args.availability_zone, datastore=parsed_args.datastore, datastore_version=parsed_args.datastore_version, datastore_version_number=parsed_args.datastore_version_number, nics=nics, configuration=parsed_args.configuration, replica_of=replica_of, replica_count=replica_count, modules=modules, locality=locality, region_name=parsed_args.region, access=access)
        instance = set_attributes_for_print_detail(instance)
        return zip(*sorted(instance.items()))