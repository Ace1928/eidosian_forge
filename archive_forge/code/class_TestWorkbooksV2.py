import os.path
import pkg_resources as pkg
from urllib import parse
from urllib import request
from mistralclient.api import base as api_base
from mistralclient.api.v2 import workbooks
from mistralclient.tests.unit.v2 import base
class TestWorkbooksV2(base.BaseClientV2Test):

    def test_create(self):
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE, json=WORKBOOK, status_code=201)
        wb = self.workbooks.create(WB_DEF)
        self.assertIsNotNone(wb)
        self.assertEqual(WB_DEF, wb.definition)
        last_request = self.requests_mock.last_request
        self.assertEqual(WB_DEF, last_request.text)
        self.assertEqual('text/plain', last_request.headers['content-type'])

    def test_create_with_file_uri(self):
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE, json=WORKBOOK, status_code=201)
        path = pkg.resource_filename('mistralclient', 'tests/unit/resources/wb_v2.yaml')
        path = os.path.abspath(path)
        uri = parse.urljoin('file:', request.pathname2url(path))
        wb = self.workbooks.create(uri)
        self.assertIsNotNone(wb)
        self.assertEqual(WB_DEF, wb.definition)
        last_request = self.requests_mock.last_request
        self.assertEqual(WB_DEF, last_request.text)
        self.assertEqual('text/plain', last_request.headers['content-type'])

    def test_update(self):
        self.requests_mock.put(self.TEST_URL + URL_TEMPLATE, json=WORKBOOK)
        wb = self.workbooks.update(WB_DEF)
        self.assertIsNotNone(wb)
        self.assertEqual(WB_DEF, wb.definition)
        last_request = self.requests_mock.last_request
        self.assertEqual(WB_DEF, last_request.text)
        self.assertEqual('text/plain', last_request.headers['content-type'])

    def test_update_with_file(self):
        self.requests_mock.put(self.TEST_URL + URL_TEMPLATE, json=WORKBOOK)
        path = pkg.resource_filename('mistralclient', 'tests/unit/resources/wb_v2.yaml')
        wb = self.workbooks.update(path)
        self.assertIsNotNone(wb)
        self.assertEqual(WB_DEF, wb.definition)
        last_request = self.requests_mock.last_request
        self.assertEqual(WB_DEF, last_request.text)
        self.assertEqual('text/plain', last_request.headers['content-type'])

    def test_list(self):
        self.requests_mock.get(self.TEST_URL + URL_TEMPLATE, json={'workbooks': [WORKBOOK]})
        workbook_list = self.workbooks.list()
        self.assertEqual(1, len(workbook_list))
        wb = workbook_list[0]
        self.assertEqual(workbooks.Workbook(self.workbooks, WORKBOOK).to_dict(), wb.to_dict())

    def test_get(self):
        self.requests_mock.get(self.TEST_URL + URL_TEMPLATE_NAME % 'wb', json=WORKBOOK)
        wb = self.workbooks.get('wb')
        self.assertIsNotNone(wb)
        self.assertEqual(workbooks.Workbook(self.workbooks, WORKBOOK).to_dict(), wb.to_dict())

    def test_delete(self):
        url = self.TEST_URL + URL_TEMPLATE_NAME % 'wb'
        self.requests_mock.delete(url, status_code=204)
        self.workbooks.delete('wb')

    def test_validate(self):
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE_VALIDATE, json={'valid': True})
        result = self.workbooks.validate(WB_DEF)
        self.assertIsNotNone(result)
        self.assertIn('valid', result)
        self.assertTrue(result['valid'])
        last_request = self.requests_mock.last_request
        self.assertEqual(WB_DEF, last_request.text)
        self.assertEqual('text/plain', last_request.headers['content-type'])

    def test_validate_with_file(self):
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE_VALIDATE, json={'valid': True})
        path = pkg.resource_filename('mistralclient', 'tests/unit/resources/wb_v2.yaml')
        result = self.workbooks.validate(path)
        self.assertIsNotNone(result)
        self.assertIn('valid', result)
        self.assertTrue(result['valid'])
        last_request = self.requests_mock.last_request
        self.assertEqual(WB_DEF, last_request.text)
        self.assertEqual('text/plain', last_request.headers['content-type'])

    def test_validate_failed(self):
        mock_result = {'valid': False, 'error': "Task properties 'action' and 'workflow' can't be specified both"}
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE_VALIDATE, json=mock_result)
        result = self.workbooks.validate(INVALID_WB_DEF)
        self.assertIsNotNone(result)
        self.assertIn('valid', result)
        self.assertFalse(result['valid'])
        self.assertIn('error', result)
        self.assertIn("Task properties 'action' and 'workflow' can't be specified both", result['error'])
        last_request = self.requests_mock.last_request
        self.assertEqual(INVALID_WB_DEF, last_request.text)
        self.assertEqual('text/plain', last_request.headers['content-type'])

    def test_validate_api_failed(self):
        self.requests_mock.post(self.TEST_URL + URL_TEMPLATE_VALIDATE, status_code=500)
        self.assertRaises(api_base.APIException, self.workbooks.validate, WB_DEF)
        last_request = self.requests_mock.last_request
        self.assertEqual(WB_DEF, last_request.text)
        self.assertEqual('text/plain', last_request.headers['content-type'])