from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils
class UpdateNodeGroup(command.Command):
    _description = _('Update a Nodegroup')

    def get_parser(self, prog_name):
        parser = super(UpdateNodeGroup, self).get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help=_('ID or name of the cluster where the nodegroup belongs.'))
        parser.add_argument('nodegroup', metavar='<nodegroup>', help=_('The name or UUID of cluster to update'))
        parser.add_argument('op', metavar='<op>', choices=['add', 'replace', 'remove'], help=_("Operations: one of 'add', 'replace' or 'remove'"))
        parser.add_argument('attributes', metavar='<path=value>', nargs='+', action='append', default=[], help=_('Attributes to add/replace or remove (only PATH is necessary on remove)'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        patch = magnum_utils.args_array_to_patch(parsed_args.op, parsed_args.attributes[0])
        cluster_id = parsed_args.cluster
        mag_client.nodegroups.update(cluster_id, parsed_args.nodegroup, patch)
        print('Request to update nodegroup %s has been accepted.' % parsed_args.nodegroup)