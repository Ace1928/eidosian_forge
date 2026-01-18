from __future__ import division
import json
import os
import pickle
import collections
import contextlib
import warnings
import re
import io
import requests
import pytest
import urllib3
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth, _basic_auth_str
from requests.compat import (
from requests.cookies import (
from requests.exceptions import (
from requests.exceptions import SSLError as RequestsSSLError
from requests.models import PreparedRequest
from requests.structures import CaseInsensitiveDict
from requests.sessions import SessionRedirectMixin
from requests.models import urlencode
from requests.hooks import default_hooks
from requests.compat import JSONDecodeError, is_py3, MutableMapping
from .compat import StringIO, u
from .utils import override_environ
from urllib3.util import Timeout as Urllib3Timeout
class TestPreparingURLs(object):

    @pytest.mark.parametrize('url,expected', (('http://google.com', 'http://google.com/'), (u'http://ジェーピーニック.jp', u'http://xn--hckqz9bzb1cyrb.jp/'), (u'http://xn--n3h.net/', u'http://xn--n3h.net/'), (u'http://ジェーピーニック.jp'.encode('utf-8'), u'http://xn--hckqz9bzb1cyrb.jp/'), (u'http://straße.de/straße', u'http://xn--strae-oqa.de/stra%C3%9Fe'), (u'http://straße.de/straße'.encode('utf-8'), u'http://xn--strae-oqa.de/stra%C3%9Fe'), (u'http://Königsgäßchen.de/straße', u'http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe'), (u'http://Königsgäßchen.de/straße'.encode('utf-8'), u'http://xn--knigsgchen-b4a3dun.de/stra%C3%9Fe'), (b'http://xn--n3h.net/', u'http://xn--n3h.net/'), (b'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/', u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/'), (u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/', u'http://[1200:0000:ab00:1234:0000:2552:7777:1313]:12345/')))
    def test_preparing_url(self, url, expected):

        def normalize_percent_encode(x):
            for c in re.findall('%[a-fA-F0-9]{2}', x):
                x = x.replace(c, c.upper())
            return x
        r = requests.Request('GET', url=url)
        p = r.prepare()
        assert normalize_percent_encode(p.url) == expected

    @pytest.mark.parametrize('url', (b'http://*.google.com', b'http://*', u'http://*.google.com', u'http://*', u'http://☃.net/'))
    def test_preparing_bad_url(self, url):
        r = requests.Request('GET', url=url)
        with pytest.raises(requests.exceptions.InvalidURL):
            r.prepare()

    @pytest.mark.parametrize('url, exception', (('http://localhost:-1', InvalidURL),))
    def test_redirecting_to_bad_url(self, httpbin, url, exception):
        with pytest.raises(exception):
            r = requests.get(httpbin('redirect-to'), params={'url': url})

    @pytest.mark.parametrize('input, expected', ((b'http+unix://%2Fvar%2Frun%2Fsocket/path%7E', u'http+unix://%2Fvar%2Frun%2Fsocket/path~'), (u'http+unix://%2Fvar%2Frun%2Fsocket/path%7E', u'http+unix://%2Fvar%2Frun%2Fsocket/path~'), (b'mailto:user@example.org', u'mailto:user@example.org'), (u'mailto:user@example.org', u'mailto:user@example.org'), (b'data:SSDimaUgUHl0aG9uIQ==', u'data:SSDimaUgUHl0aG9uIQ==')))
    def test_url_mutation(self, input, expected):
        """
        This test validates that we correctly exclude some URLs from
        preparation, and that we handle others. Specifically, it tests that
        any URL whose scheme doesn't begin with "http" is left alone, and
        those whose scheme *does* begin with "http" are mutated.
        """
        r = requests.Request('GET', url=input)
        p = r.prepare()
        assert p.url == expected

    @pytest.mark.parametrize('input, params, expected', ((b'http+unix://%2Fvar%2Frun%2Fsocket/path', {'key': 'value'}, u'http+unix://%2Fvar%2Frun%2Fsocket/path?key=value'), (u'http+unix://%2Fvar%2Frun%2Fsocket/path', {'key': 'value'}, u'http+unix://%2Fvar%2Frun%2Fsocket/path?key=value'), (b'mailto:user@example.org', {'key': 'value'}, u'mailto:user@example.org'), (u'mailto:user@example.org', {'key': 'value'}, u'mailto:user@example.org')))
    def test_parameters_for_nonstandard_schemes(self, input, params, expected):
        """
        Setting parameters for nonstandard schemes is allowed if those schemes
        begin with "http", and is forbidden otherwise.
        """
        r = requests.Request('GET', url=input, params=params)
        p = r.prepare()
        assert p.url == expected

    def test_post_json_nan(self, httpbin):
        data = {'foo': float('nan')}
        with pytest.raises(requests.exceptions.InvalidJSONError):
            r = requests.post(httpbin('post'), json=data)

    def test_json_decode_compatibility(self, httpbin):
        r = requests.get(httpbin('bytes/20'))
        with pytest.raises(requests.exceptions.JSONDecodeError) as excinfo:
            r.json()
        assert isinstance(excinfo.value, RequestException)
        assert isinstance(excinfo.value, JSONDecodeError)
        assert r.text not in str(excinfo.value)

    @pytest.mark.skipif(not is_py3, reason='doc attribute is only present on py3')
    def test_json_decode_persists_doc_attr(self, httpbin):
        r = requests.get(httpbin('bytes/20'))
        with pytest.raises(requests.exceptions.JSONDecodeError) as excinfo:
            r.json()
        assert excinfo.value.doc == r.text