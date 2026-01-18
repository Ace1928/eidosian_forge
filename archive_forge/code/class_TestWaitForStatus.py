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
class TestWaitForStatus(TestWait):

    def test_immediate_status(self):
        status = 'loling'
        res = mock.Mock(spec=['id', 'status'])
        res.status = status
        result = resource.wait_for_status(self.cloud.compute, res, status, None, interval=1, wait=1)
        self.assertEqual(res, result)

    def test_immediate_status_case(self):
        status = 'LOLing'
        res = mock.Mock(spec=['id', 'status'])
        res.status = status
        result = resource.wait_for_status(self.cloud.compute, res, 'lOling', None, interval=1, wait=1)
        self.assertEqual(res, result)

    def test_immediate_status_different_attribute(self):
        status = 'loling'
        res = mock.Mock(spec=['id', 'mood'])
        res.mood = status
        result = resource.wait_for_status(self.cloud.compute, res, status, None, interval=1, wait=1, attribute='mood')
        self.assertEqual(res, result)

    def test_status_match(self):
        status = 'loling'
        statuses = ['first', 'other', 'another', 'another', status]
        res = self._fake_resource(statuses)
        result = resource.wait_for_status(mock.Mock(), res, status, None, interval=1, wait=5)
        self.assertEqual(result, res)

    def test_status_match_with_none(self):
        status = 'loling'
        statuses = [None, 'other', None, 'another', status]
        res = self._fake_resource(statuses)
        result = resource.wait_for_status(mock.Mock(), res, status, None, interval=1, wait=5)
        self.assertEqual(result, res)

    def test_status_match_none(self):
        status = None
        statuses = ['first', 'other', 'another', 'another', status]
        res = self._fake_resource(statuses)
        result = resource.wait_for_status(mock.Mock(), res, status, None, interval=1, wait=5)
        self.assertEqual(result, res)

    def test_status_match_different_attribute(self):
        status = 'loling'
        statuses = ['first', 'other', 'another', 'another', status]
        res = self._fake_resource(statuses, attribute='mood')
        result = resource.wait_for_status(mock.Mock(), res, status, None, interval=1, wait=5, attribute='mood')
        self.assertEqual(result, res)

    def test_status_fails(self):
        failure = 'crying'
        statuses = ['success', 'other', failure]
        res = self._fake_resource(statuses)
        self.assertRaises(exceptions.ResourceFailure, resource.wait_for_status, mock.Mock(), res, 'loling', [failure], interval=1, wait=5)

    def test_status_fails_different_attribute(self):
        failure = 'crying'
        statuses = ['success', 'other', failure]
        res = self._fake_resource(statuses, attribute='mood')
        self.assertRaises(exceptions.ResourceFailure, resource.wait_for_status, mock.Mock(), res, 'loling', [failure.upper()], interval=1, wait=5, attribute='mood')

    def test_timeout(self):
        status = 'loling'
        statuses = ['other'] * 7
        res = self._fake_resource(statuses)
        self.assertRaises(exceptions.ResourceTimeout, resource.wait_for_status, self.cloud.compute, res, status, None, 0.01, 0.1)

    def test_no_sleep(self):
        statuses = ['other']
        res = self._fake_resource(statuses)
        self.assertRaises(exceptions.ResourceTimeout, resource.wait_for_status, self.cloud.compute, res, 'status', None, interval=0, wait=-1)

    def test_callback(self):
        """Callback is called with 'progress' attribute."""
        statuses = ['building', 'building', 'building', 'building', 'active']
        progresses = [0, 25, 50, 100]
        res = self._fake_resource(statuses=statuses, progresses=progresses)
        callback = mock.Mock()
        result = resource.wait_for_status(mock.Mock(), res, 'active', None, interval=0.1, wait=1, callback=callback)
        self.assertEqual(result, res)
        callback.assert_has_calls([mock.call(x) for x in progresses])

    def test_callback_without_progress(self):
        """Callback is called with 0 if 'progress' attribute is missing."""
        statuses = ['building', 'building', 'building', 'building', 'active']
        res = self._fake_resource(statuses=statuses)
        callback = mock.Mock()
        result = resource.wait_for_status(mock.Mock(), res, 'active', None, interval=0.1, wait=1, callback=callback)
        self.assertEqual(result, res)
        callback.assert_has_calls([mock.call(0)] * 3)