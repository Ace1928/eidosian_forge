from collections import defaultdict
from collections import namedtuple
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from openstack import connection
from oslo_config import cfg
from oslo_log import log
from oslo_limit import exception
from oslo_limit import opts
def _get_project_limit(self, project_id, resource_name):
    if project_id in self.plimit_cache and resource_name in self.plimit_cache[project_id]:
        return self.plimit_cache[project_id][resource_name]
    limits = self.connection.limits(service_id=self._service_id, region_id=self._region_id, resource_name=resource_name, project_id=project_id)
    try:
        limit = next(limits)
    except StopIteration:
        return None
    if self.should_cache and limit:
        self.plimit_cache[project_id][resource_name] = limit
    return limit