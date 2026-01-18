from collections import defaultdict
from collections import namedtuple
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from openstack import connection
from oslo_config import cfg
from oslo_log import log
from oslo_limit import exception
from oslo_limit import opts
class _EnforcerUtils(object):
    """Logic common used by multiple enforcers"""

    def __init__(self, cache=True):
        self.connection = _get_keystone_connection()
        self.should_cache = cache
        self.plimit_cache = defaultdict(dict)
        self.rlimit_cache = {}
        endpoint_id = CONF.oslo_limit.endpoint_id
        if not endpoint_id:
            raise ValueError('endpoint_id is not configured')
        self._endpoint = self.connection.get_endpoint(endpoint_id)
        if not self._endpoint:
            raise ValueError("can't find endpoint for %s" % endpoint_id)
        self._service_id = self._endpoint.service_id
        self._region_id = self._endpoint.region_id

    @staticmethod
    def enforce_limits(project_id, limits, current_usage, deltas):
        """Check that proposed usage is not over given limits

        :param project_id: project being checked or None
        :param limits: list of (resource_name,limit) pairs
        :param current_usage: dict of resource name and current usage
        :param deltas: dict of resource name and proposed additional usage

        :raises exception.ClaimExceedsLimit: raise if over limit
        """
        over_limit_list = []
        for resource_name, limit in limits:
            if resource_name not in current_usage:
                msg = 'unable to get current usage for %s' % resource_name
                raise ValueError(msg)
            current = int(current_usage[resource_name])
            delta = int(deltas[resource_name])
            proposed_usage_total = current + delta
            if proposed_usage_total > limit:
                over_limit_list.append(exception.OverLimitInfo(resource_name, limit, current, delta))
        if len(over_limit_list) > 0:
            LOG.debug('hit limit for project: %s', over_limit_list)
            raise exception.ProjectOverLimit(project_id, over_limit_list)

    def get_registered_limits(self, resource_names):
        """Get all the default limits for a given resource name list

        :param resource_names: list of resource_name strings
        :return: list of (resource_name, limit) pairs
        """
        registered_limits = []
        for resource_name in resource_names:
            reg_limit = self._get_registered_limit(resource_name)
            if reg_limit:
                limit = reg_limit.default_limit
            else:
                limit = 0
            registered_limits.append((resource_name, limit))
        return registered_limits

    def get_project_limits(self, project_id, resource_names):
        """Get all the limits for given project a resource_name list

        If a limit is not found, it will be considered to be zero
        (i.e. no quota)

        :param project_id: project being checked or None
        :param resource_names: list of resource_name strings
        :return: list of (resource_name,limit) pairs
        """
        project_limits = []
        for resource_name in resource_names:
            try:
                limit = self._get_limit(project_id, resource_name)
            except _LimitNotFound:
                limit = 0
            project_limits.append((resource_name, limit))
        return project_limits

    def _get_limit(self, project_id, resource_name):
        project_limit = self._get_project_limit(project_id, resource_name) if project_id is not None else None
        if project_limit:
            return project_limit.resource_limit
        registered_limit = self._get_registered_limit(resource_name)
        if registered_limit:
            return registered_limit.default_limit
        LOG.error('Unable to find registered limit for resource %(resource)s for %(service)s in region %(region)s.', {'resource': resource_name, 'service': self._service_id, 'region': self._region_id}, exec_info=False)
        raise _LimitNotFound(resource_name)

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

    def _get_registered_limit(self, resource_name):
        if resource_name in self.rlimit_cache:
            return self.rlimit_cache[resource_name]
        reg_limits = self.connection.registered_limits(service_id=self._service_id, region_id=self._region_id, resource_name=resource_name)
        try:
            reg_limit = next(reg_limits)
        except StopIteration:
            return None
        if self.should_cache and reg_limit:
            self.rlimit_cache[resource_name] = reg_limit
        return reg_limit