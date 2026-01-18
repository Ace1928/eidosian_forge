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
class TestWaitForDelete(TestWait):

    def test_success_not_found(self):
        response = mock.Mock()
        response.headers = {}
        response.status_code = 404
        res = mock.Mock()
        res.fetch.side_effect = [res, res, exceptions.ResourceNotFound('Not Found', response)]
        result = resource.wait_for_delete(self.cloud.compute, res, 1, 3)
        self.assertEqual(result, res)

    def test_status(self):
        """Successful deletion indicated by status."""
        statuses = ['active', 'deleting', 'deleting', 'deleting', 'deleted']
        res = self._fake_resource(statuses=statuses)
        result = resource.wait_for_delete(mock.Mock(), res, interval=0.1, wait=1)
        self.assertEqual(result, res)

    def test_callback(self):
        """Callback is called with 'progress' attribute."""
        statuses = ['active', 'deleting', 'deleting', 'deleting', 'deleted']
        progresses = [0, 25, 50, 100]
        res = self._fake_resource(statuses=statuses, progresses=progresses)
        callback = mock.Mock()
        result = resource.wait_for_delete(mock.Mock(), res, interval=1, wait=5, callback=callback)
        self.assertEqual(result, res)
        callback.assert_has_calls([mock.call(x) for x in progresses])

    def test_callback_without_progress(self):
        """Callback is called with 0 if 'progress' attribute is missing."""
        statuses = ['active', 'deleting', 'deleting', 'deleting', 'deleted']
        res = self._fake_resource(statuses=statuses)
        callback = mock.Mock()
        result = resource.wait_for_delete(mock.Mock(), res, interval=1, wait=5, callback=callback)
        self.assertEqual(result, res)
        callback.assert_has_calls([mock.call(0)] * 3)

    def test_timeout(self):
        res = mock.Mock()
        res.status = 'ACTIVE'
        res.fetch.return_value = res
        self.assertRaises(exceptions.ResourceTimeout, resource.wait_for_delete, self.cloud.compute, res, 0.1, 0.3)