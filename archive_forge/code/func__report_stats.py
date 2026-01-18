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
def _report_stats(self, response, url=None, method=None, exc=None):
    if self._statsd_client:
        self._report_stats_statsd(response, url, method, exc)
    if self._prometheus_counter and self._prometheus_histogram:
        self._report_stats_prometheus(response, url, method, exc)
    if self._influxdb_client:
        self._report_stats_influxdb(response, url, method, exc)