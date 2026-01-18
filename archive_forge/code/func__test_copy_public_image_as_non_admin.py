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
def _test_copy_public_image_as_non_admin(self):
    self.start_servers(**self.__dict__.copy())
    image_id = self._create_and_import_image_data()
    path = self._url('/v2/images/%s' % image_id)
    response = requests.get(path, headers=self._headers())
    self.assertEqual(http.OK, response.status_code)
    self.assertEqual('file1', jsonutils.loads(response.text)['stores'])
    path = self._url('/v2/images/%s/import' % image_id)
    headers = self._headers({'content-type': 'application/json'})
    headers = get_auth_header(TENANT2, TENANT2, role='reader,member', headers=headers)
    data = jsonutils.dumps({'method': {'name': 'copy-image'}, 'stores': ['file2']})
    response = requests.post(path, headers=headers, data=data)
    return (image_id, response)