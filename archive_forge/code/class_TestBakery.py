from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
class TestBakery(TestCase):

    def assert_cookie_security(self, cookies, name, secure):
        for cookie in cookies:
            if cookie.name == name:
                assert cookie.secure == secure
                break
        else:
            assert False, 'no cookie named {} found in jar'.format(name)

    def test_discharge(self):
        client = httpbakery.Client()
        with HTTMock(first_407_then_200), HTTMock(discharge_200):
            resp = requests.get(ID_PATH, cookies=client.cookies, auth=client.auth())
        resp.raise_for_status()
        assert 'macaroon-test' in client.cookies.keys()
        self.assert_cookie_security(client.cookies, 'macaroon-test', secure=False)

    @patch('webbrowser.open')
    def test_407_then_401_on_discharge(self, mock_open):
        client = httpbakery.Client()
        with HTTMock(first_407_then_200), HTTMock(discharge_401), HTTMock(wait_after_401):
            resp = requests.get(ID_PATH, cookies=client.cookies, auth=client.auth())
            resp.raise_for_status()
        mock_open.assert_called_once_with(u'http://example.com/visit', new=1)
        assert 'macaroon-test' in client.cookies.keys()

    @patch('webbrowser.open')
    def test_407_then_error_on_wait(self, mock_open):
        client = httpbakery.Client()
        with HTTMock(first_407_then_200), HTTMock(discharge_401), HTTMock(wait_on_error):
            with self.assertRaises(httpbakery.InteractionError) as exc:
                requests.get(ID_PATH, cookies=client.cookies, auth=client.auth())
        self.assertEqual(str(exc.exception), 'cannot start interactive session: cannot get http://example.com/wait')
        mock_open.assert_called_once_with(u'http://example.com/visit', new=1)

    def test_407_then_no_interaction_methods(self):
        client = httpbakery.Client(interaction_methods=[])
        with HTTMock(first_407_then_200), HTTMock(discharge_401):
            with self.assertRaises(httpbakery.InteractionError) as exc:
                requests.get(ID_PATH, cookies=client.cookies, auth=client.auth())
        self.assertEqual(str(exc.exception), 'cannot start interactive session: interaction required but not possible')

    def test_407_then_unknown_interaction_methods(self):

        class UnknownInteractor(httpbakery.Interactor):

            def kind(self):
                return 'unknown'
        client = httpbakery.Client(interaction_methods=[UnknownInteractor()])
        with HTTMock(first_407_then_200), HTTMock(discharge_401), HTTMock(visit_200):
            with self.assertRaises(httpbakery.InteractionError) as exc:
                requests.get(ID_PATH, cookies=client.cookies, auth=client.auth())
        self.assertEqual(str(exc.exception), 'cannot start interactive session: no methods supported; supported [unknown]; provided [interactive]')

    def test_cookie_with_port(self):
        client = httpbakery.Client()
        with HTTMock(first_407_then_200_with_port):
            with HTTMock(discharge_200):
                resp = requests.get('http://example.com:8000/someprotecteurl', cookies=client.cookies, auth=client.auth())
        resp.raise_for_status()
        assert 'macaroon-test' in client.cookies.keys()

    def test_secure_cookie_for_https(self):
        client = httpbakery.Client()
        with HTTMock(first_407_then_200_with_port), HTTMock(discharge_200):
            resp = requests.get('https://example.com:8000/someprotecteurl', cookies=client.cookies, auth=client.auth())
        resp.raise_for_status()
        assert 'macaroon-test' in client.cookies.keys()
        self.assert_cookie_security(client.cookies, 'macaroon-test', secure=True)