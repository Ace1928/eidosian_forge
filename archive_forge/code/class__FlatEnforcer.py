from collections import defaultdict
from collections import namedtuple
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from openstack import connection
from oslo_config import cfg
from oslo_log import log
from oslo_limit import exception
from oslo_limit import opts
class _FlatEnforcer(object):
    name = 'flat'

    def __init__(self, usage_callback, cache=True):
        self._usage_callback = usage_callback
        self._utils = _EnforcerUtils(cache=cache)

    def get_registered_limits(self, resources_to_check):
        return self._utils.get_registered_limits(resources_to_check)

    def get_project_limits(self, project_id, resources_to_check):
        return self._utils.get_project_limits(project_id, resources_to_check)

    def get_project_usage(self, project_id, resources_to_check):
        return self._usage_callback(project_id, resources_to_check)

    def enforce(self, project_id, deltas):
        resources_to_check = list(deltas.keys())
        resources_to_check.sort()
        project_limits = self.get_project_limits(project_id, resources_to_check)
        current_usage = self.get_project_usage(project_id, resources_to_check)
        self._utils.enforce_limits(project_id, project_limits, current_usage, deltas)