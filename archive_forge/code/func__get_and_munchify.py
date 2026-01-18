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
def _get_and_munchify(self, key, data):
    """Wrapper around meta.get_and_munchify.

        Some of the methods expect a `meta` attribute to be passed in as
        part of the method signature. In those methods the meta param is
        overriding the meta module making the call to meta.get_and_munchify
        to fail.
        """
    if isinstance(data, requests.models.Response):
        data = proxy._json_response(data)
    return meta.get_and_munchify(key, data)