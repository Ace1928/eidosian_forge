import sys
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from saharaclient.osc import utils
from saharaclient.osc.v1 import clusters as c_v1
class ScaleCluster(c_v1.ScaleCluster):
    """Scales cluster"""
    log = logging.getLogger(__name__ + '.ScaleCluster')

    def _get_json_arg_helptext(self):
        return '\n               JSON representation of the cluster scale object. Other\n               arguments (except for --wait) will not be taken into\n               account if this one is provided. Specifiying a JSON\n               object is also the only way to indicate specific\n               instances to decomission.\n               '

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.data_processing
        data = self._take_action(client, parsed_args)
        _format_cluster_output(self.app, data)
        data = utils.prepare_data(data, c_v1.CLUSTER_FIELDS)
        return self.dict2columns(data)