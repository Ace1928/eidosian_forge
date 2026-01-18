from oslo_log import log as logging
from saharaclient.osc.v1 import data_sources as ds_v1
class CreateDataSource(ds_v1.CreateDataSource):
    """Creates data source"""
    log = logging.getLogger(__name__ + '.CreateDataSource')