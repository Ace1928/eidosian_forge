from osc_lib import utils as osc_utils
from oslo_log import log as logging
from saharaclient.osc import utils
from saharaclient.osc.v1 import cluster_templates as ct_v1
class ExportClusterTemplate(ct_v1.ExportClusterTemplate):
    """Export cluster template to JSON"""
    log = logging.getLogger(__name__ + '.ExportClusterTemplate')