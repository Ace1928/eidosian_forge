import io
import http.client
import json as jsonutils
from requests.adapters import HTTPAdapter
from requests.cookies import MockRequest, MockResponse
from requests.cookies import RequestsCookieJar
from requests.cookies import merge_cookies, cookiejar_from_dict
from requests.utils import get_encoding_from_headers
from urllib3.response import HTTPResponse
from requests_mock import exceptions
class _MatcherResponse(object):

    def __init__(self, **kwargs):
        self._exc = kwargs.pop('exc', None)
        if self._exc and kwargs:
            raise TypeError('Cannot provide other arguments with exc.')
        _check_body_arguments(**kwargs)
        self._params = kwargs
        content = self._params.get('content')
        text = self._params.get('text')
        if content is not None and (not (callable(content) or isinstance(content, bytes))):
            raise TypeError('Content should be a callback or binary data')
        if text is not None and (not (callable(text) or isinstance(text, str))):
            raise TypeError('Text should be a callback or string data')

    def get_response(self, request):
        if self._exc:
            raise self._exc
        cookies = self._params.get('cookies', CookieJar())
        if isinstance(cookies, dict):
            cookies = cookiejar_from_dict(cookies, CookieJar())
        context = _Context(self._params.get('headers', {}).copy(), self._params.get('status_code', _DEFAULT_STATUS), self._params.get('reason'), cookies)

        def _call(f, *args, **kwargs):
            return f(request, context, *args, **kwargs) if callable(f) else f
        return create_response(request, json=_call(self._params.get('json')), text=_call(self._params.get('text')), content=_call(self._params.get('content')), body=_call(self._params.get('body')), raw=_call(self._params.get('raw')), json_encoder=self._params.get('json_encoder'), status_code=context.status_code, reason=context.reason, headers=context.headers, cookies=context.cookies)