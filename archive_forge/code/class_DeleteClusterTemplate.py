from magnumclient.common import utils as magnum_utils
from magnumclient.exceptions import InvalidAttribute
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_log import log as logging
class DeleteClusterTemplate(command.Command):
    """Delete a Cluster Template."""
    _description = _('Delete a Cluster Template.')

    def get_parser(self, prog_name):
        parser = super(DeleteClusterTemplate, self).get_parser(prog_name)
        parser.add_argument('cluster-templates', metavar='<cluster-templates>', nargs='+', help=_('ID or name of the (cluster template)s to delete.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        for cluster_template in getattr(parsed_args, 'cluster-templates'):
            try:
                mag_client.cluster_templates.delete(cluster_template)
                print('Request to delete cluster template %s has been accepted.' % cluster_template)
            except Exception as e:
                print('Delete for cluster template %(cluster_template)s failed: %(e)s' % {'cluster_template': cluster_template, 'e': e})