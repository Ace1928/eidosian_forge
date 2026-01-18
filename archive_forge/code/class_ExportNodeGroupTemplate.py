import sys
from osc_lib import utils as osc_utils
from saharaclient.osc import utils
from saharaclient.osc.v1 import node_group_templates as ngt_v1
class ExportNodeGroupTemplate(ngt_v1.ExportNodeGroupTemplate, utils.NodeGroupTemplatesUtils):
    """Export node group template to JSON"""

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.data_processing
        self._export_take_action(client, parsed_args)