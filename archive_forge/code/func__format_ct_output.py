from osc_lib import utils as osc_utils
from oslo_log import log as logging
from saharaclient.osc import utils
from saharaclient.osc.v1 import cluster_templates as ct_v1
def _format_ct_output(app, data):
    data['node_groups'] = ct_v1._format_node_groups_list(data['node_groups'])
    data['anti_affinity'] = osc_utils.format_list(data['anti_affinity'])