from oslo_log import log as logging
from saharaclient.osc.v1 import data_sources as ds_v1
class ShowDataSource(ds_v1.ShowDataSource):
    """Display data source details"""
    log = logging.getLogger(__name__ + '.ShowDataSource')