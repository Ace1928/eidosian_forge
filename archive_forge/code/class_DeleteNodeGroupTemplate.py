import sys
from osc_lib import utils as osc_utils
from saharaclient.osc import utils
from saharaclient.osc.v1 import node_group_templates as ngt_v1
class DeleteNodeGroupTemplate(ngt_v1.DeleteNodeGroupTemplate, utils.NodeGroupTemplatesUtils):
    """Deletes node group template"""

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.data_processing
        for ngt in parsed_args.node_group_template:
            ngt_id = utils.get_resource_id(client.node_group_templates, ngt)
            client.node_group_templates.delete(ngt_id)
            sys.stdout.write('Node group template "{ngt}" has been removed successfully.\n'.format(ngt=ngt))