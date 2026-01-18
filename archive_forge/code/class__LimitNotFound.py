from collections import defaultdict
from collections import namedtuple
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from openstack import connection
from oslo_config import cfg
from oslo_log import log
from oslo_limit import exception
from oslo_limit import opts
class _LimitNotFound(Exception):

    def __init__(self, resource):
        msg = "Can't find the limit for resource %(resource)s" % {'resource': resource}
        self.resource = resource
        super(_LimitNotFound, self).__init__(msg)