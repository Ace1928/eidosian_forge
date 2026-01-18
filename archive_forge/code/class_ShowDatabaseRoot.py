from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from troveclient.i18n import _
class ShowDatabaseRoot(command.ShowOne):
    _description = _('Gets status if root was ever enabled for an instance or cluster.')

    def get_parser(self, prog_name):
        parser = super(ShowDatabaseRoot, self).get_parser(prog_name)
        parser.add_argument('instance_or_cluster', metavar='<instance_or_cluster>', help=_('ID or name of the instance or cluster.'))
        return parser

    def take_action(self, parsed_args):
        database_client_manager = self.app.client_manager.database
        instance_or_cluster, resource_type = find_instance_or_cluster(database_client_manager, parsed_args.instance_or_cluster)
        db_root = database_client_manager.root
        if resource_type == 'instance':
            root = db_root.is_instance_root_enabled(instance_or_cluster)
        else:
            root = db_root.is_cluster_root_enabled(instance_or_cluster)
        result = {'is_root_enabled': root.rootEnabled}
        return zip(*sorted(result.items()))