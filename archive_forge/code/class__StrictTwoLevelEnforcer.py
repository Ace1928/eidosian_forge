from collections import defaultdict
from collections import namedtuple
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from openstack import connection
from oslo_config import cfg
from oslo_log import log
from oslo_limit import exception
from oslo_limit import opts
class _StrictTwoLevelEnforcer(object):
    name = 'strict-two-level'

    def __init__(self, usage_callback, cache=True):
        self._usage_callback = usage_callback

    def get_registered_limits(self, resources_to_check):
        raise NotImplementedError()

    def get_project_limits(self, project_id, resources_to_check):
        raise NotImplementedError()

    def get_project_usage(self, project_id, resources_to_check):
        raise NotImplementedError()

    def enforce(self, project_id, deltas):
        raise NotImplementedError()