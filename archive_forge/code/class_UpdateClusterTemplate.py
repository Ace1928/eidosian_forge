from magnumclient.common import utils as magnum_utils
from magnumclient.exceptions import InvalidAttribute
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_log import log as logging
class UpdateClusterTemplate(command.ShowOne):
    """Update a Cluster Template."""
    log = logging.getLogger(__name__ + '.UpdateClusterTemplate')

    def get_parser(self, prog_name):
        parser = super(UpdateClusterTemplate, self).get_parser(prog_name)
        parser.add_argument('cluster-template', metavar='<cluster-template>', help=_('The name or UUID of cluster template to update'))
        parser.add_argument('op', metavar='<op>', choices=['add', 'replace', 'remove'], help=_("Operations: one of 'add', 'replace' or 'remove'"))
        parser.add_argument('attributes', metavar='<path=value>', nargs='+', action='append', default=[], help=_('Attributes to add/replace or remove (only PATH is necessary on remove)'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        patch = magnum_utils.args_array_to_patch(parsed_args.op, parsed_args.attributes[0])
        name = getattr(parsed_args, 'cluster-template', None)
        ct = mag_client.cluster_templates.update(name, patch)
        return _show_cluster_template(ct)