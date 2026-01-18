import contextlib
import functools
import sys
import threading
import types
import requests
from requests_mock import adapter
from requests_mock import exceptions
class MockerCore(object):
    """A wrapper around common mocking functions.

    Automate the process of mocking the requests library. This will keep the
    same general options available and prevent repeating code.
    """
    _PROXY_FUNCS = {'last_request', 'add_matcher', 'request_history', 'called', 'called_once', 'call_count', 'reset'}
    case_sensitive = False
    "case_sensitive handles a backwards incompatible bug. The URL used to\n    match against our matches and that is saved in request_history is always\n    lowercased. This is incorrect as it reports incorrect history to the user\n    and doesn't allow case sensitive path matching.\n\n    Unfortunately fixing this change is backwards incompatible in the 1.X\n    series as people may rely on this behaviour. To work around this you can\n    globally set:\n\n    requests_mock.mock.case_sensitive = True\n\n    or for pytest set in your configuration:\n\n    [pytest]\n    requests_mock_case_sensitive = True\n\n    which will prevent the lowercase being executed and return case sensitive\n    url and query information.\n\n    This will become the default in a 2.X release. See bug: #1584008.\n    "

    def __init__(self, session=None, **kwargs):
        if session and (not isinstance(session, requests.Session)):
            raise TypeError('Only a requests.Session object can be mocked')
        self._mock_target = session or requests.Session
        self.case_sensitive = kwargs.pop('case_sensitive', self.case_sensitive)
        self._adapter = kwargs.pop('adapter', None) or adapter.Adapter(case_sensitive=self.case_sensitive)
        self._json_encoder = kwargs.pop('json_encoder', None)
        self.real_http = kwargs.pop('real_http', False)
        self._last_send = None
        if kwargs:
            raise TypeError('Unexpected Arguments: %s' % ', '.join(kwargs))

    def start(self):
        """Start mocking requests.

        Install the adapter and the wrappers required to intercept requests.
        """
        if self._last_send:
            raise RuntimeError('Mocker has already been started')
        self._last_send = self._mock_target.send
        self._last_get_adapter = self._mock_target.get_adapter

        def _fake_get_adapter(session, url):
            return self._adapter

        def _fake_send(session, request, **kwargs):
            with threading_rlock(timeout=10):
                _set_method(session, 'get_adapter', _fake_get_adapter)
                try:
                    return _original_send(session, request, **kwargs)
                except exceptions.NoMockAddress:
                    if not self.real_http:
                        raise
                except adapter._RunRealHTTP:
                    pass
                finally:
                    _set_method(session, 'get_adapter', self._last_get_adapter)
            if isinstance(self._mock_target, type):
                return self._last_send(session, request, **kwargs)
            else:
                return self._last_send(request, **kwargs)
        _set_method(self._mock_target, 'send', _fake_send)

    def stop(self):
        """Stop mocking requests.

        This should have no impact if mocking has not been started.
        When nesting mockers, make sure to stop the innermost first.
        """
        if self._last_send:
            self._mock_target.send = self._last_send
            self._last_send = None

    def reset_mock(self):
        self.reset()

    def __getattr__(self, name):
        if name in self._PROXY_FUNCS:
            try:
                return getattr(self._adapter, name)
            except AttributeError:
                pass
        raise AttributeError(name)

    def register_uri(self, *args, **kwargs):
        kwargs['_real_http'] = kwargs.pop('real_http', False)
        kwargs.setdefault('json_encoder', self._json_encoder)
        return self._adapter.register_uri(*args, **kwargs)

    def request(self, *args, **kwargs):
        return self.register_uri(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self.request(GET, *args, **kwargs)

    def options(self, *args, **kwargs):
        return self.request(OPTIONS, *args, **kwargs)

    def head(self, *args, **kwargs):
        return self.request(HEAD, *args, **kwargs)

    def post(self, *args, **kwargs):
        return self.request(POST, *args, **kwargs)

    def put(self, *args, **kwargs):
        return self.request(PUT, *args, **kwargs)

    def patch(self, *args, **kwargs):
        return self.request(PATCH, *args, **kwargs)

    def delete(self, *args, **kwargs):
        return self.request(DELETE, *args, **kwargs)