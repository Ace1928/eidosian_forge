from oslo_serialization import jsonutils as json
from saharaclient.api import job_binary_internals as jbi
from saharaclient.tests.unit import base
class JobBinaryInternalTest(base.BaseTestCase):
    body = {'name': 'name', 'datasize': '123', 'id': 'id'}

    def test_create_job_binary_internal(self):
        url = self.URL + '/job-binary-internals/name'
        self.responses.put(url, status_code=202, json={'job_binary_internal': self.body})
        resp = self.client.job_binary_internals.create('name', 'data')
        self.assertEqual(url, self.responses.last_request.url)
        self.assertEqual('data', self.responses.last_request.body)
        self.assertIsInstance(resp, jbi.JobBinaryInternal)
        self.assertFields(self.body, resp)

    def test_job_binary_internal_list(self):
        url = self.URL + '/job-binary-internals'
        self.responses.get(url, json={'binaries': [self.body]})
        resp = self.client.job_binary_internals.list()
        self.assertEqual(url, self.responses.last_request.url)
        self.assertIsInstance(resp[0], jbi.JobBinaryInternal)
        self.assertFields(self.body, resp[0])

    def test_job_binary_get(self):
        url = self.URL + '/job-binary-internals/id'
        self.responses.get(url, json={'job_binary_internal': self.body})
        resp = self.client.job_binary_internals.get('id')
        self.assertEqual(url, self.responses.last_request.url)
        self.assertIsInstance(resp, jbi.JobBinaryInternal)
        self.assertFields(self.body, resp)

    def test_job_binary_delete(self):
        url = self.URL + '/job-binary-internals/id'
        self.responses.delete(url, status_code=204)
        self.client.job_binary_internals.delete('id')
        self.assertEqual(url, self.responses.last_request.url)

    def test_job_binary_update(self):
        url = self.URL + '/job-binary-internals/id'
        update_body = {'name': 'new_name'}
        self.responses.patch(url, status_code=202, json=update_body)
        resp = self.client.job_binary_internals.update('id', name='new_name')
        self.assertEqual(url, self.responses.last_request.url)
        self.assertIsInstance(resp, jbi.JobBinaryInternal)
        self.assertEqual(update_body, json.loads(self.responses.last_request.body))
        self.client.job_binary_internals.update('id')
        self.assertEqual(url, self.responses.last_request.url)
        self.assertEqual({}, json.loads(self.responses.last_request.body))
        unset_json = {'name': None, 'is_public': None, 'is_protected': None}
        self.client.job_binary_internals.update('id', **unset_json)
        self.assertEqual(url, self.responses.last_request.url)
        self.assertEqual(unset_json, json.loads(self.responses.last_request.body))