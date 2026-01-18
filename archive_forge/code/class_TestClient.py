import base64
import datetime
import json
import platform
import threading
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery.httpbakery as httpbakery
import pymacaroons
import requests
import macaroonbakery._utils as utils
from macaroonbakery.httpbakery._error import DischargeError
from fixtures import (
from httmock import HTTMock, urlmatch
from six.moves.urllib.parse import parse_qs
from six.moves.urllib.request import Request
class TestClient(TestWithFixtures):

    def setUp(self):
        super(TestClient, self).setUp()
        self.useFixture(EnvironmentVariable('http_proxy'))
        self.useFixture(EnvironmentVariable('HTTP_PROXY'))

    def test_single_service_first_party(self):
        b = new_bakery('loc', None, None)

        def handler(*args):
            GetHandler(b, None, None, None, None, AGES, *args)
        try:
            httpd = HTTPServer(('', 0), handler)
            thread = threading.Thread(target=httpd.serve_forever)
            thread.start()
            srv_macaroon = b.oven.macaroon(version=bakery.LATEST_VERSION, expiry=AGES, caveats=None, ops=[TEST_OP])
            self.assertEqual(srv_macaroon.macaroon.location, 'loc')
            client = httpbakery.Client()
            client.cookies.set_cookie(requests.cookies.create_cookie('macaroon-test', base64.b64encode(json.dumps([srv_macaroon.to_dict().get('m')]).encode('utf-8')).decode('utf-8')))
            resp = requests.get(url='http://' + httpd.server_address[0] + ':' + str(httpd.server_address[1]), cookies=client.cookies, auth=client.auth())
            resp.raise_for_status()
            self.assertEqual(resp.text, 'done')
        finally:
            httpd.shutdown()

    def test_single_service_third_party(self):

        class _DischargerLocator(bakery.ThirdPartyLocator):

            def __init__(self):
                self.key = bakery.generate_key()

            def third_party_info(self, loc):
                if loc == 'http://1.2.3.4':
                    return bakery.ThirdPartyInfo(public_key=self.key.public_key, version=bakery.LATEST_VERSION)
        d = _DischargerLocator()
        b = new_bakery('loc', d, None)

        @urlmatch(path='.*/discharge')
        def discharge(url, request):
            self.assertEqual(url.path, '/discharge')
            qs = parse_qs(request.body)
            content = {q: qs[q][0] for q in qs}
            m = httpbakery.discharge(checkers.AuthContext(), content, d.key, d, alwaysOK3rd)
            return {'status_code': 200, 'content': {'Macaroon': m.to_dict()}}

        def handler(*args):
            GetHandler(b, 'http://1.2.3.4', None, None, None, AGES, *args)
        try:
            httpd = HTTPServer(('', 0), handler)
            server_url = 'http://' + httpd.server_address[0] + ':' + str(httpd.server_address[1])
            thread = threading.Thread(target=httpd.serve_forever)
            thread.start()
            client = httpbakery.Client()
            with HTTMock(discharge):
                resp = requests.get(url=server_url, cookies=client.cookies, auth=client.auth())
            resp.raise_for_status()
            self.assertEqual(resp.text, 'done')
        finally:
            httpd.shutdown()

    def test_single_service_third_party_with_path(self):

        class _DischargerLocator(bakery.ThirdPartyLocator):

            def __init__(self):
                self.key = bakery.generate_key()

            def third_party_info(self, loc):
                if loc == 'http://1.2.3.4/some/path':
                    return bakery.ThirdPartyInfo(public_key=self.key.public_key, version=bakery.LATEST_VERSION)
        d = _DischargerLocator()
        b = new_bakery('loc', d, None)

        @urlmatch(path='.*/discharge')
        def discharge(url, request):
            self.assertEqual(url.path, '/some/path/discharge')
            qs = parse_qs(request.body)
            content = {q: qs[q][0] for q in qs}
            m = httpbakery.discharge(checkers.AuthContext(), content, d.key, d, alwaysOK3rd)
            return {'status_code': 200, 'content': {'Macaroon': m.to_dict()}}

        def handler(*args):
            GetHandler(b, 'http://1.2.3.4/some/path', None, None, None, AGES, *args)
        try:
            httpd = HTTPServer(('', 0), handler)
            server_url = 'http://' + httpd.server_address[0] + ':' + str(httpd.server_address[1])
            thread = threading.Thread(target=httpd.serve_forever)
            thread.start()
            client = httpbakery.Client()
            with HTTMock(discharge):
                resp = requests.get(url=server_url, cookies=client.cookies, auth=client.auth())
            resp.raise_for_status()
            self.assertEqual(resp.text, 'done')
        finally:
            httpd.shutdown()

    def test_single_service_third_party_version_1_caveat(self):

        class _DischargerLocator(bakery.ThirdPartyLocator):

            def __init__(self):
                self.key = bakery.generate_key()

            def third_party_info(self, loc):
                if loc == 'http://1.2.3.4':
                    return bakery.ThirdPartyInfo(public_key=self.key.public_key, version=bakery.VERSION_1)
        d = _DischargerLocator()
        b = new_bakery('loc', d, None)

        @urlmatch(path='.*/discharge')
        def discharge(url, request):
            qs = parse_qs(request.body)
            content = {q: qs[q][0] for q in qs}
            m = httpbakery.discharge(checkers.AuthContext(), content, d.key, d, alwaysOK3rd)
            return {'status_code': 200, 'content': {'Macaroon': m.to_dict()}}

        def handler(*args):
            GetHandler(b, 'http://1.2.3.4', None, None, None, AGES, *args)
        try:
            httpd = HTTPServer(('', 0), handler)
            server_url = 'http://' + httpd.server_address[0] + ':' + str(httpd.server_address[1])
            thread = threading.Thread(target=httpd.serve_forever)
            thread.start()
            client = httpbakery.Client()
            with HTTMock(discharge):
                resp = requests.get(url=server_url, cookies=client.cookies, auth=client.auth())
            resp.raise_for_status()
            self.assertEqual(resp.text, 'done')
        finally:
            httpd.shutdown()

    def test_cookie_domain_host_not_fqdn(self):
        b = new_bakery('loc', None, None)

        def handler(*args):
            GetHandler(b, None, None, None, None, AGES, *args)
        try:
            httpd = HTTPServer(('', 0), handler)
            thread = threading.Thread(target=httpd.serve_forever)
            thread.start()
            srv_macaroon = b.oven.macaroon(version=bakery.LATEST_VERSION, expiry=AGES, caveats=None, ops=[TEST_OP])
            self.assertEqual(srv_macaroon.macaroon.location, 'loc')
            client = httpbakery.Client()
            resp = requests.get(url='http://localhost:' + str(httpd.server_address[1]), cookies=client.cookies, auth=client.auth())
            resp.raise_for_status()
            self.assertEqual(resp.text, 'done')
        except httpbakery.BakeryException:
            pass
        finally:
            httpd.shutdown()
        [cookie] = client.cookies
        self.assertEqual(cookie.name, 'macaroon-test')
        self.assertEqual(cookie.domain, 'localhost.local')

    def test_single_party_with_header(self):
        b = new_bakery('loc', None, None)

        def handler(*args):
            GetHandler(b, None, None, None, None, AGES, *args)
        try:
            httpd = HTTPServer(('', 0), handler)
            thread = threading.Thread(target=httpd.serve_forever)
            thread.start()
            srv_macaroon = b.oven.macaroon(version=bakery.LATEST_VERSION, expiry=AGES, caveats=None, ops=[TEST_OP])
            self.assertEqual(srv_macaroon.macaroon.location, 'loc')
            headers = {'Macaroons': base64.b64encode(json.dumps([srv_macaroon.to_dict().get('m')]).encode('utf-8'))}
            resp = requests.get(url='http://' + httpd.server_address[0] + ':' + str(httpd.server_address[1]), headers=headers)
            resp.raise_for_status()
            self.assertEqual(resp.text, 'done')
        finally:
            httpd.shutdown()

    def test_expiry_cookie_is_set(self):

        class _DischargerLocator(bakery.ThirdPartyLocator):

            def __init__(self):
                self.key = bakery.generate_key()

            def third_party_info(self, loc):
                if loc == 'http://1.2.3.4':
                    return bakery.ThirdPartyInfo(public_key=self.key.public_key, version=bakery.LATEST_VERSION)
        d = _DischargerLocator()
        b = new_bakery('loc', d, None)

        @urlmatch(path='.*/discharge')
        def discharge(url, request):
            qs = parse_qs(request.body)
            content = {q: qs[q][0] for q in qs}
            m = httpbakery.discharge(checkers.AuthContext(), content, d.key, d, alwaysOK3rd)
            return {'status_code': 200, 'content': {'Macaroon': m.to_dict()}}
        ages = datetime.datetime.utcnow() + datetime.timedelta(days=1)

        def handler(*args):
            GetHandler(b, 'http://1.2.3.4', None, None, None, ages, *args)
        try:
            httpd = HTTPServer(('', 0), handler)
            thread = threading.Thread(target=httpd.serve_forever)
            thread.start()
            client = httpbakery.Client()
            with HTTMock(discharge):
                resp = requests.get(url='http://' + httpd.server_address[0] + ':' + str(httpd.server_address[1]), cookies=client.cookies, auth=client.auth())
            resp.raise_for_status()
            m = bakery.Macaroon.from_dict(json.loads(base64.b64decode(client.cookies.get('macaroon-test')).decode('utf-8'))[0])
            t = checkers.macaroons_expiry_time(checkers.Namespace(), [m.macaroon])
            self.assertEqual(ages, t)
            self.assertEqual(resp.text, 'done')
        finally:
            httpd.shutdown()

    def test_expiry_cookie_set_in_past(self):

        class _DischargerLocator(bakery.ThirdPartyLocator):

            def __init__(self):
                self.key = bakery.generate_key()

            def third_party_info(self, loc):
                if loc == 'http://1.2.3.4':
                    return bakery.ThirdPartyInfo(public_key=self.key.public_key, version=bakery.LATEST_VERSION)
        d = _DischargerLocator()
        b = new_bakery('loc', d, None)

        @urlmatch(path='.*/discharge')
        def discharge(url, request):
            qs = parse_qs(request.body)
            content = {q: qs[q][0] for q in qs}
            m = httpbakery.discharge(checkers.AuthContext(), content, d.key, d, alwaysOK3rd)
            return {'status_code': 200, 'content': {'Macaroon': m.to_dict()}}
        ages = datetime.datetime.utcnow() - datetime.timedelta(days=1)

        def handler(*args):
            GetHandler(b, 'http://1.2.3.4', None, None, None, ages, *args)
        try:
            httpd = HTTPServer(('', 0), handler)
            thread = threading.Thread(target=httpd.serve_forever)
            thread.start()
            client = httpbakery.Client()
            with HTTMock(discharge):
                with self.assertRaises(httpbakery.BakeryException) as ctx:
                    requests.get(url='http://' + httpd.server_address[0] + ':' + str(httpd.server_address[1]), cookies=client.cookies, auth=client.auth())
            self.assertEqual(ctx.exception.args[0], 'too many (3) discharge requests')
        finally:
            httpd.shutdown()

    def test_too_many_discharge(self):

        class _DischargerLocator(bakery.ThirdPartyLocator):

            def __init__(self):
                self.key = bakery.generate_key()

            def third_party_info(self, loc):
                if loc == 'http://1.2.3.4':
                    return bakery.ThirdPartyInfo(public_key=self.key.public_key, version=bakery.LATEST_VERSION)
        d = _DischargerLocator()
        b = new_bakery('loc', d, None)

        @urlmatch(path='.*/discharge')
        def discharge(url, request):
            wrong_macaroon = bakery.Macaroon(root_key=b'some key', id=b'xxx', location='some other location', version=bakery.VERSION_0)
            return {'status_code': 200, 'content': {'Macaroon': wrong_macaroon.to_dict()}}

        def handler(*args):
            GetHandler(b, 'http://1.2.3.4', None, None, None, AGES, *args)
        try:
            httpd = HTTPServer(('', 0), handler)
            thread = threading.Thread(target=httpd.serve_forever)
            thread.start()
            client = httpbakery.Client()
            with HTTMock(discharge):
                with self.assertRaises(httpbakery.BakeryException) as ctx:
                    requests.get(url='http://' + httpd.server_address[0] + ':' + str(httpd.server_address[1]), cookies=client.cookies, auth=client.auth())
            self.assertEqual(ctx.exception.args[0], 'too many (3) discharge requests')
        finally:
            httpd.shutdown()

    def test_third_party_discharge_refused(self):

        class _DischargerLocator(bakery.ThirdPartyLocator):

            def __init__(self):
                self.key = bakery.generate_key()

            def third_party_info(self, loc):
                if loc == 'http://1.2.3.4':
                    return bakery.ThirdPartyInfo(public_key=self.key.public_key, version=bakery.LATEST_VERSION)

        def check(cond, arg):
            raise bakery.ThirdPartyCaveatCheckFailed('boo! cond' + cond)
        d = _DischargerLocator()
        b = new_bakery('loc', d, None)

        @urlmatch(path='.*/discharge')
        def discharge(url, request):
            qs = parse_qs(request.body)
            content = {q: qs[q][0] for q in qs}
            httpbakery.discharge(checkers.AuthContext(), content, d.key, d, ThirdPartyCaveatCheckerF(check))

        def handler(*args):
            GetHandler(b, 'http://1.2.3.4', None, None, None, AGES, *args)
        try:
            httpd = HTTPServer(('', 0), handler)
            thread = threading.Thread(target=httpd.serve_forever)
            thread.start()
            client = httpbakery.Client()
            with HTTMock(discharge):
                with self.assertRaises(bakery.ThirdPartyCaveatCheckFailed):
                    requests.get(url='http://' + httpd.server_address[0] + ':' + str(httpd.server_address[1]), cookies=client.cookies, auth=client.auth())
        finally:
            httpd.shutdown()

    def test_discharge_with_interaction_required_error(self):

        class _DischargerLocator(bakery.ThirdPartyLocator):

            def __init__(self):
                self.key = bakery.generate_key()

            def third_party_info(self, loc):
                if loc == 'http://1.2.3.4':
                    return bakery.ThirdPartyInfo(public_key=self.key.public_key, version=bakery.LATEST_VERSION)
        d = _DischargerLocator()
        b = new_bakery('loc', d, None)

        @urlmatch(path='.*/discharge')
        def discharge(url, request):
            return {'status_code': 401, 'content': {'Code': httpbakery.ERR_INTERACTION_REQUIRED, 'Message': 'interaction required', 'Info': {'WaitURL': 'http://0.1.2.3/', 'VisitURL': 'http://0.1.2.3/'}}}

        def handler(*args):
            GetHandler(b, 'http://1.2.3.4', None, None, None, AGES, *args)
        try:
            httpd = HTTPServer(('', 0), handler)
            thread = threading.Thread(target=httpd.serve_forever)
            thread.start()

            class MyInteractor(httpbakery.LegacyInteractor):

                def legacy_interact(self, ctx, location, visit_url):
                    raise httpbakery.InteractionError('cannot visit')

                def interact(self, ctx, location, interaction_required_err):
                    pass

                def kind(self):
                    return httpbakery.WEB_BROWSER_INTERACTION_KIND
            client = httpbakery.Client(interaction_methods=[MyInteractor()])
            with HTTMock(discharge):
                with self.assertRaises(httpbakery.InteractionError):
                    requests.get('http://' + httpd.server_address[0] + ':' + str(httpd.server_address[1]), cookies=client.cookies, auth=client.auth())
        finally:
            httpd.shutdown()

    def test_discharge_jsondecodeerror(self):

        class _DischargerLocator(bakery.ThirdPartyLocator):

            def __init__(self):
                self.key = bakery.generate_key()

            def third_party_info(self, loc):
                if loc == 'http://1.2.3.4':
                    return bakery.ThirdPartyInfo(public_key=self.key.public_key, version=bakery.LATEST_VERSION)
        d = _DischargerLocator()
        b = new_bakery('loc', d, None)

        @urlmatch(path='.*/discharge')
        def discharge(url, request):
            return {'status_code': 503, 'content': 'bad system'}

        def handler(*args):
            GetHandler(b, 'http://1.2.3.4', None, None, None, AGES, *args)
        try:
            httpd = HTTPServer(('', 0), handler)
            thread = threading.Thread(target=httpd.serve_forever)
            thread.start()
            client = httpbakery.Client()
            with HTTMock(discharge):
                with self.assertRaises(DischargeError) as discharge_error:
                    requests.get('http://' + httpd.server_address[0] + ':' + str(httpd.server_address[1]), cookies=client.cookies, auth=client.auth())
                if platform.python_version_tuple()[0] == '2':
                    self.assertEqual(str(discharge_error.exception), "third party refused dischargex: unexpected response: [503] 'bad system'")
                else:
                    self.assertEqual(str(discharge_error.exception), "third party refused dischargex: unexpected response: [503] b'bad system'")
        finally:
            httpd.shutdown()

    def test_extract_macaroons_from_request(self):

        def encode_macaroon(m):
            macaroons = '[' + utils.macaroon_to_json_string(m) + ']'
            return base64.urlsafe_b64encode(utils.to_bytes(macaroons)).decode('ascii')
        req = Request('http://example.com')
        m1 = pymacaroons.Macaroon(version=pymacaroons.MACAROON_V2, identifier='one')
        req.add_header('Macaroons', encode_macaroon(m1))
        m2 = pymacaroons.Macaroon(version=pymacaroons.MACAROON_V2, identifier='two')
        jar = requests.cookies.RequestsCookieJar()
        jar.set_cookie(utils.cookie(name='macaroon-auth', value=encode_macaroon(m2), url='http://example.com'))
        jar.set_cookie(utils.cookie(name='macaroon-empty', value='', url='http://example.com'))
        jar.add_cookie_header(req)
        macaroons = httpbakery.extract_macaroons(req)
        self.assertEqual(len(macaroons), 2)
        macaroons.sort(key=lambda ms: ms[0].identifier)
        self.assertEqual(macaroons[0][0].identifier, m1.identifier)
        self.assertEqual(macaroons[1][0].identifier, m2.identifier)

    def test_handle_error_cookie_path(self):
        macaroon = bakery.Macaroon(root_key=b'some key', id=b'xxx', location='some location', version=bakery.VERSION_0)
        info = {'Macaroon': macaroon.to_dict(), 'MacaroonPath': '.', 'CookieNameSuffix': 'test'}
        error = httpbakery.Error(code=407, message='error', version=bakery.LATEST_VERSION, info=httpbakery.ErrorInfo.from_dict(info))
        client = httpbakery.Client()
        client.handle_error(error, 'http://example.com/some/path')
        [cookie] = client.cookies
        self.assertEqual(cookie.path, '/some/')