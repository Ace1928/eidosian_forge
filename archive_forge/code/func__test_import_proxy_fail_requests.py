import hashlib
import http.client as http
import os
import subprocess
import tempfile
import time
import urllib
import uuid
import fixtures
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_serialization import jsonutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional import ft_utils as func_utils
from glance.tests import utils as test_utils
def _test_import_proxy_fail_requests(self, error, status):
    self.ksa_client.return_value.post.side_effect = error
    self.ksa_client.return_value.delete.side_effect = error
    self.config(worker_self_reference_url='http://worker1')
    self.start_server(set_worker_url=False)
    image_id = self._create_and_stage()
    self.config(worker_self_reference_url='http://worker2')
    self.start_server(set_worker_url=False)
    r = self._import_direct(image_id, ['store1'])
    self.assertEqual(status, r.status)
    self.assertIn(b'Stage host is unavailable', r.body)
    r = self.api_delete('/v2/images/%s' % image_id)
    self.assertEqual(204, r.status_code)
    r = self.api_get('/v2/images/%s' % image_id)
    self.assertEqual(404, r.status_code)