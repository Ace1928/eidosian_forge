from oslo_serialization import jsonutils
from mistralclient.api import base as api_base
from mistralclient.api.v2 import executions
from mistralclient.tests.unit.v2 import base
class TestExecutionsV2(base.BaseClientV2Test):

    def test_create(self):
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE, json=EXEC, status_code=201)
        body = {'workflow_name': EXEC['workflow_name'], 'description': '', 'input': jsonutils.dumps(EXEC['input'])}
        ex = self.executions.create(EXEC['workflow_name'], EXEC['workflow_namespace'], EXEC['input'])
        self.assertIsNotNone(ex)
        self.assertDictEqual(executions.Execution(self.executions, EXEC).to_dict(), ex.to_dict())
        self.assertDictEqual(body, self.requests_mock.last_request.json())

    def test_create_with_workflow_id(self):
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE, json=EXEC, status_code=201)
        body = {'workflow_id': EXEC['workflow_id'], 'description': '', 'input': jsonutils.dumps(EXEC['input'])}
        ex = self.executions.create(EXEC['workflow_id'], workflow_input=EXEC['input'])
        self.assertIsNotNone(ex)
        self.assertDictEqual(executions.Execution(self.executions, EXEC).to_dict(), ex.to_dict())
        self.assertDictEqual(body, self.requests_mock.last_request.json())

    def test_create_with_source_execution_id(self):
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE, json=SOURCE_EXEC, status_code=201)
        body = {'description': '', 'source_execution_id': SOURCE_EXEC['source_execution_id']}
        ex = self.executions.create(source_execution_id=SOURCE_EXEC['source_execution_id'])
        self.assertIsNotNone(ex)
        self.assertDictEqual(executions.Execution(self.executions, SOURCE_EXEC).to_dict(), ex.to_dict())
        self.assertDictEqual(body, self.requests_mock.last_request.json())

    def test_create_failure1(self):
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE, json=EXEC, status_code=201)
        self.assertRaises(api_base.APIException, self.executions.create, '')

    def test_update(self):
        url = self.TEST_URL + URL_TEMPLATE_ID % EXEC['id']
        self.requests_mock.put(url, json=EXEC)
        body = {'state': EXEC['state']}
        ex = self.executions.update(EXEC['id'], EXEC['state'])
        self.assertIsNotNone(ex)
        self.assertDictEqual(executions.Execution(self.executions, EXEC).to_dict(), ex.to_dict())
        self.assertDictEqual(body, self.requests_mock.last_request.json())

    def test_update_env(self):
        url = self.TEST_URL + URL_TEMPLATE_ID % EXEC['id']
        self.requests_mock.put(url, json=EXEC)
        body = {'state': EXEC['state'], 'params': {'env': {'k1': 'foobar'}}}
        ex = self.executions.update(EXEC['id'], EXEC['state'], env={'k1': 'foobar'})
        self.assertIsNotNone(ex)
        self.assertDictEqual(executions.Execution(self.executions, EXEC).to_dict(), ex.to_dict())
        self.assertDictEqual(body, self.requests_mock.last_request.json())

    def test_list(self):
        self.requests_mock.get(self.TEST_URL + URL_TEMPLATE, json={'executions': [EXEC, SUB_WF_EXEC]})
        execution_list = self.executions.list()
        self.assertEqual(2, len(execution_list))
        self.assertDictEqual(executions.Execution(self.executions, EXEC).to_dict(), execution_list[0].to_dict())
        self.assertDictEqual(executions.Execution(self.executions, SUB_WF_EXEC).to_dict(), execution_list[1].to_dict())

    def test_list_with_pagination(self):
        self.requests_mock.get(self.TEST_URL + URL_TEMPLATE, json={'executions': [EXEC], 'next': '/executions?fake'})
        execution_list = self.executions.list(limit=1, sort_keys='created_at', sort_dirs='asc')
        self.assertEqual(1, len(execution_list))
        last_request = self.requests_mock.last_request
        self.assertEqual(['1'], last_request.qs['limit'])
        self.assertEqual(['created_at'], last_request.qs['sort_keys'])
        self.assertEqual(['asc'], last_request.qs['sort_dirs'])

    def test_list_with_no_limit(self):
        self.requests_mock.get(self.TEST_URL + URL_TEMPLATE, json={'executions': [EXEC]})
        execution_list = self.executions.list(limit=-1)
        self.assertEqual(1, len(execution_list))
        last_request = self.requests_mock.last_request
        self.assertNotIn('limit', last_request.qs)

    def test_get(self):
        url = self.TEST_URL + URL_TEMPLATE_ID % EXEC['id']
        self.requests_mock.get(url, json=EXEC)
        ex = self.executions.get(EXEC['id'])
        self.assertDictEqual(executions.Execution(self.executions, EXEC).to_dict(), ex.to_dict())

    def test_get_sub_wf_ex(self):
        url = self.TEST_URL + URL_TEMPLATE_ID % SUB_WF_EXEC['id']
        self.requests_mock.get(url, json=SUB_WF_EXEC)
        ex = self.executions.get(SUB_WF_EXEC['id'])
        self.assertDictEqual(executions.Execution(self.executions, SUB_WF_EXEC).to_dict(), ex.to_dict())

    def test_delete_with_force(self):
        url = self.TEST_URL + URL_TEMPLATE_ID % EXEC['id']
        self.requests_mock.delete(url, status_code=204)
        self.executions.delete(EXEC['id'], force=True)

    def test_delete(self):
        url = self.TEST_URL + URL_TEMPLATE_ID % EXEC['id']
        self.requests_mock.delete(url, status_code=204)
        self.executions.delete(EXEC['id'])

    def test_report_statistics_only(self):
        url = self.TEST_URL + URL_TEMPLATE_ID % EXEC['id'] + '/report?statistics_only=True'
        expected_json = {'statistics': {}}
        self.requests_mock.get(url, json=expected_json)
        report = self.executions.get_report(EXEC['id'], statistics_only=True)
        self.assertDictEqual(expected_json, report)

    def test_report(self):
        url = self.TEST_URL + URL_TEMPLATE_ID % EXEC['id'] + '/report'
        expected_json = {'root_workflow_execution': {}, 'statistics': {}}
        self.requests_mock.get(url, json=expected_json)
        report = self.executions.get_report(EXEC['id'])
        self.assertDictEqual(expected_json, report)

    def test_get_sub_executions(self):
        url = self.TEST_URL + URL_TEMPLATE_SUB_EXECUTIONS % (EXEC['id'], '?max_depth=-1&errors_only=')
        self.requests_mock.get(url, json={'executions': [EXEC, SUB_WF_EXEC]})
        sub_execution_list = self.executions.get_ex_sub_executions(EXEC['id'])
        self.assertEqual(2, len(sub_execution_list))
        self.assertDictEqual(executions.Execution(self.executions, EXEC).to_dict(), sub_execution_list[0].to_dict())
        self.assertDictEqual(executions.Execution(self.executions, SUB_WF_EXEC).to_dict(), sub_execution_list[1].to_dict())