import collections
import os
import tempfile
import time
import urllib
import uuid
import fixtures
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from requests import structures
from requests_mock.contrib import fixture as rm_fixture
import openstack.cloud
import openstack.config as occ
import openstack.connection
from openstack.fixture import connection as os_fixture
from openstack.tests import base
from openstack.tests import fakes
def __do_register_uris(self, uri_mock_list=None):
    for to_mock in uri_mock_list:
        kw_params = {k: to_mock.pop(k) for k in ('request_headers', 'complete_qs', '_real_http') if k in to_mock}
        method = to_mock.pop('method')
        uri = to_mock.pop('uri')
        key = '{method}|{uri}|{params}'.format(method=method, uri=uri, params=kw_params)
        validate = to_mock.pop('validate', {})
        valid_keys = set(['json', 'headers', 'params', 'data'])
        invalid_keys = set(validate.keys()) - valid_keys
        if invalid_keys:
            raise TypeError('Invalid values passed to validate: {keys}'.format(keys=invalid_keys))
        headers = structures.CaseInsensitiveDict(to_mock.pop('headers', {}))
        if 'content-type' not in headers:
            headers[u'content-type'] = 'application/json'
        if 'exc' not in to_mock:
            to_mock['headers'] = headers
        self.calls += [dict(method=method, url=uri, **validate)]
        self._uri_registry.setdefault(key, {'response_list': [], 'kw_params': kw_params})
        if self._uri_registry[key]['kw_params'] != kw_params:
            raise AssertionError('PROGRAMMING ERROR: key-word-params should be part of the uri_key and cannot change, it will affect the matcher in requests_mock. %(old)r != %(new)r' % {'old': self._uri_registry[key]['kw_params'], 'new': kw_params})
        self._uri_registry[key]['response_list'].append(to_mock)
    for mocked, params in self._uri_registry.items():
        mock_method, mock_uri, _ignored = mocked.split('|', 2)
        self.adapter.register_uri(mock_method, mock_uri, params['response_list'], **params['kw_params'])