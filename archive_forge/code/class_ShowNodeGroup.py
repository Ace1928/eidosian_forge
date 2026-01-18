from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils
class ShowNodeGroup(command.ShowOne):
    _description = _('Show a nodegroup')

    def get_parser(self, prog_name):
        parser = super(ShowNodeGroup, self).get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help=_('ID or name of the cluster where the nodegroup belongs.'))
        parser.add_argument('nodegroup', metavar='<nodegroup>', help=_('ID or name of the nodegroup to show.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        columns = NODEGROUP_ATTRIBUTES
        mag_client = self.app.client_manager.container_infra
        cluster_id = parsed_args.cluster
        nodegroup = mag_client.nodegroups.get(cluster_id, parsed_args.nodegroup)
        return (columns, utils.get_item_properties(nodegroup, columns))