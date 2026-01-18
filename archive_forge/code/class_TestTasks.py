import http.client as http
import uuid
from oslo_serialization import jsonutils
import requests
from glance.tests import functional
class TestTasks(functional.FunctionalTest):

    def setUp(self):
        super(TestTasks, self).setUp()
        self.cleanup()
        self.api_server.deployment_flavor = 'noauth'

    def _headers(self, custom_headers=None):
        base_headers = {'X-Identity-Status': 'Confirmed', 'X-Auth-Token': '932c5c84-02ac-4fe5-a9ba-620af0e2bb96', 'X-User-Id': 'f9a41d13-0c13-47e9-bee2-ce4e8bfe958e', 'X-Tenant-Id': TENANT1, 'X-Roles': 'admin'}
        base_headers.update(custom_headers or {})
        return base_headers

    def test_task_not_allowed_non_admin(self):
        self.start_servers(**self.__dict__.copy())
        roles = {'X-Roles': 'member'}
        path = self._url('/v2/tasks')
        response = requests.get(path, headers=self._headers(roles))
        self.assertEqual(http.FORBIDDEN, response.status_code)

    def test_task_lifecycle(self):
        self.start_servers(**self.__dict__.copy())
        path = self._url('/v2/tasks')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tasks = jsonutils.loads(response.text)['tasks']
        self.assertEqual(0, len(tasks))
        path = self._url('/v2/tasks')
        headers = self._headers({'content-type': 'application/json'})
        data = jsonutils.dumps({'type': 'import', 'input': {'import_from': 'http://example.com', 'import_from_format': 'qcow2', 'image_properties': {'disk_format': 'vhd', 'container_format': 'ovf'}}})
        response = requests.post(path, headers=headers, data=data)
        self.assertEqual(http.CREATED, response.status_code)
        task = jsonutils.loads(response.text)
        task_id = task['id']
        self.assertIn('Location', response.headers)
        self.assertEqual(path + '/' + task_id, response.headers['Location'])
        checked_keys = set(['created_at', 'id', 'input', 'message', 'owner', 'schema', 'self', 'status', 'type', 'result', 'updated_at', 'request_id', 'user_id'])
        self.assertEqual(checked_keys, set(task.keys()))
        expected_task = {'status': 'pending', 'type': 'import', 'input': {'import_from': 'http://example.com', 'import_from_format': 'qcow2', 'image_properties': {'disk_format': 'vhd', 'container_format': 'ovf'}}, 'schema': '/v2/schemas/task'}
        for key, value in expected_task.items():
            self.assertEqual(value, task[key], key)
        path = self._url('/v2/tasks')
        response = requests.get(path, headers=self._headers())
        self.assertEqual(http.OK, response.status_code)
        tasks = jsonutils.loads(response.text)['tasks']
        self.assertEqual(1, len(tasks))
        self.assertEqual(task_id, tasks[0]['id'])
        path = self._url('/v2/tasks/%s' % tasks[0]['id'])
        response = requests.delete(path, headers=self._headers())
        self.assertEqual(http.METHOD_NOT_ALLOWED, response.status_code)
        self.assertIsNotNone(response.headers.get('Allow'))
        self.assertEqual('GET', response.headers.get('Allow'))
        self.stop_servers()