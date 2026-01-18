import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def _test_commit(self, commit_method='PUT', prepend_key=True, has_body=True, microversion=None, commit_args=None, expected_args=None, base_path=None, explicit_microversion=None):
    self.sot.commit_method = commit_method
    self.sot._body = mock.Mock()
    self.sot._body.dirty = mock.Mock(return_value={'x': 'y'})
    commit_args = commit_args or {}
    if explicit_microversion is not None:
        commit_args['microversion'] = explicit_microversion
        microversion = explicit_microversion
    self.sot.commit(self.session, prepend_key=prepend_key, has_body=has_body, base_path=base_path, **commit_args)
    self.sot._prepare_request.assert_called_once_with(prepend_key=prepend_key, base_path=base_path)
    if commit_method == 'PATCH':
        self.session.patch.assert_called_once_with(self.request.url, json=self.request.body, headers=self.request.headers, microversion=microversion, **expected_args or {})
    elif commit_method == 'POST':
        self.session.post.assert_called_once_with(self.request.url, json=self.request.body, headers=self.request.headers, microversion=microversion, **expected_args or {})
    elif commit_method == 'PUT':
        self.session.put.assert_called_once_with(self.request.url, json=self.request.body, headers=self.request.headers, microversion=microversion, **expected_args or {})
    self.assertEqual(self.sot.microversion, microversion)
    self.sot._translate_response.assert_called_once_with(self.response, has_body=has_body)