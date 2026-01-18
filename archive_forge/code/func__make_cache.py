import copy
import functools
import queue
import warnings
import dogpile.cache
import keystoneauth1.exceptions
import keystoneauth1.session
import requests.models
import requestsexceptions
from openstack import _log
from openstack.cloud import _object_store
from openstack.cloud import _utils
from openstack.cloud import meta
import openstack.config
from openstack.config import cloud_region as cloud_region_mod
from openstack import exceptions
from openstack import proxy
from openstack import utils
def _make_cache(self, cache_class, expiration_time, arguments):
    return dogpile.cache.make_region(function_key_generator=self._make_cache_key).configure(cache_class, expiration_time=expiration_time, arguments=arguments)