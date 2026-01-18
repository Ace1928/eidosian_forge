import functools
import typing as ty
import urllib
from urllib.parse import urlparse
import iso8601
import jmespath
from keystoneauth1 import adapter
from openstack import _log
from openstack import exceptions
from openstack import resource
def _report_stats_statsd(self, response, url=None, method=None, exc=None):
    try:
        if response is not None and (not url):
            url = response.request.url
        if response is not None and (not method):
            method = response.request.method
        name_parts = [normalize_metric_name(f) for f in self._extract_name(url, self.service_type, self.session.get_project_id())]
        key = '.'.join([self._statsd_prefix, normalize_metric_name(self.service_type), method, '_'.join(name_parts)])
        with self._statsd_client.pipeline() as pipe:
            if response is not None:
                duration = int(response.elapsed.total_seconds() * 1000)
                metric_name = '%s.%s' % (key, str(response.status_code))
                pipe.timing(metric_name, duration)
                pipe.incr(metric_name)
                if duration > 1000:
                    pipe.incr('%s.over_1000' % key)
            elif exc is not None:
                pipe.incr('%s.failed' % key)
            pipe.incr('%s.attempted' % key)
    except Exception:
        self.log.exception('Exception reporting metrics')