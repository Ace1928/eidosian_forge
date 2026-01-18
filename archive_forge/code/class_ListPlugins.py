from oslo_log import log as logging
from saharaclient.osc.v1 import plugins as p_v1
class ListPlugins(p_v1.ListPlugins):
    """Lists plugins"""
    log = logging.getLogger(__name__ + '.ListPlugins')