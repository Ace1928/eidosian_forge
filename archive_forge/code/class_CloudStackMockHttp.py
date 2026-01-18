import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib, urlparse, parse_qsl
from libcloud.common.types import MalformedResponseError
from libcloud.common.cloudstack import CloudStackConnection
class CloudStackMockHttp(MockHttp, unittest.TestCase):
    ERROR_TEXT = 'ERROR TEXT'

    def _response(self, status, result, response):
        return (status, json.dumps(result), {}, response)

    def _check_request(self, url):
        url = urlparse.urlparse(url)
        query = dict(parse_qsl(url.query))
        self.assertTrue('apiKey' in query)
        self.assertTrue('command' in query)
        self.assertTrue('response' in query)
        self.assertTrue('signature' in query)
        self.assertTrue(query['response'] == 'json')
        return query

    def _bad_response(self, method, url, body, headers):
        self._check_request(url)
        result = {'success': True}
        return self._response(httplib.OK, result, httplib.responses[httplib.OK])

    def _sync(self, method, url, body, headers):
        query = self._check_request(url)
        result = {query['command'].lower() + 'response': {}}
        return self._response(httplib.OK, result, httplib.responses[httplib.OK])

    def _async_success(self, method, url, body, headers):
        query = self._check_request(url)
        if query['command'].lower() == 'queryasyncjobresult':
            self.assertEqual(query['jobid'], '42')
            result = {query['command'].lower() + 'response': {'jobstatus': 1, 'jobresult': {'fake': 'result'}}}
        else:
            result = {query['command'].lower() + 'response': {'jobid': '42'}}
        return self._response(httplib.OK, result, httplib.responses[httplib.OK])

    def _async_fail(self, method, url, body, headers):
        query = self._check_request(url)
        if query['command'].lower() == 'queryasyncjobresult':
            self.assertEqual(query['jobid'], '42')
            result = {query['command'].lower() + 'response': {'jobstatus': 2, 'jobresult': {'errortext': self.ERROR_TEXT}}}
        else:
            result = {query['command'].lower() + 'response': {'jobid': '42'}}
        return self._response(httplib.OK, result, httplib.responses[httplib.OK])

    def _async_delayed(self, method, url, body, headers):
        global async_delay
        query = self._check_request(url)
        if query['command'].lower() == 'queryasyncjobresult':
            self.assertEqual(query['jobid'], '42')
            if async_delay == 0:
                result = {query['command'].lower() + 'response': {'jobstatus': 1, 'jobresult': {'fake': 'result'}}}
            else:
                result = {query['command'].lower() + 'response': {'jobstatus': 0}}
                async_delay -= 1
        else:
            result = {query['command'].lower() + 'response': {'jobid': '42'}}
        return self._response(httplib.OK, result, httplib.responses[httplib.OK])