from twisted.trial.unittest import TestCase
from twisted.web.error import UnsupportedMethod
from twisted.web.http_headers import Headers
from twisted.web.iweb import IRequest
from twisted.web.resource import (
from twisted.web.test.requesthelper import DummyRequest
def _pageRenderingTest(self, page: Resource, code: int, brief: str, detail: str) -> None:
    request = DummyRequest([b''])
    template = '\n<html>\n  <head><title>%s - %s</title></head>\n  <body>\n    <h1>%s</h1>\n    <p>%s</p>\n  </body>\n</html>\n'
    expected = template % (code, brief, brief, detail)
    self.assertEqual(page.render(request), expected.encode('utf-8'))
    self.assertEqual(request.responseCode, code)
    self.assertEqual(request.responseHeaders, Headers({b'content-type': [b'text/html; charset=utf-8']}))