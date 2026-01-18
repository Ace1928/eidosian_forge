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
def _get_cache_key_prefix(self, url):
    """Calculate cache prefix for the url"""
    name_parts = self._extract_name(url, self.service_type, self.session.get_project_id())
    return '.'.join([self.service_type] + name_parts)