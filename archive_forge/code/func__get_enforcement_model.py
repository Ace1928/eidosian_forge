from collections import defaultdict
from collections import namedtuple
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from openstack import connection
from oslo_config import cfg
from oslo_log import log
from oslo_limit import exception
from oslo_limit import opts
def _get_enforcement_model(self):
    """Query keystone for the configured enforcement model."""
    return self.connection.get('/limits/model').json()['model']['name']