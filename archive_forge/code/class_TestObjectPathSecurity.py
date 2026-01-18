import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
class TestObjectPathSecurity(PecanTestCase):

    def setUp(self):
        super(TestObjectPathSecurity, self).setUp()
        permissions_checked = set()

        class DeepSecretController(SecureController):
            authorized = False

            @expose()
            @unlocked
            def _lookup(self, someID, *remainder):
                if someID == 'notfound':
                    return None
                return (SubController(someID), remainder)

            @expose()
            def index(self):
                return 'Deep Secret'

            @classmethod
            def check_permissions(cls):
                permissions_checked.add('deepsecret')
                return cls.authorized

        class SubController(object):

            def __init__(self, myID):
                self.myID = myID

            @expose()
            def index(self):
                return 'Index %s' % self.myID
            deepsecret = DeepSecretController()

        class SecretController(SecureController):
            authorized = False
            independent_authorization = False

            @expose()
            def _lookup(self, someID, *remainder):
                if someID == 'notfound':
                    return None
                elif someID == 'lookup_wrapped':
                    return (self.wrapped, remainder)
                return (SubController(someID), remainder)

            @secure('independent_check_permissions')
            @expose()
            def independent(self):
                return 'Independent Security'
            wrapped = secure(SubController('wrapped'), 'independent_check_permissions')

            @classmethod
            def check_permissions(cls):
                permissions_checked.add('secretcontroller')
                return cls.authorized

            @classmethod
            def independent_check_permissions(cls):
                permissions_checked.add('independent')
                return cls.independent_authorization

        class NotSecretController(object):

            @expose()
            def _lookup(self, someID, *remainder):
                if someID == 'notfound':
                    return None
                return (SubController(someID), remainder)
            unlocked = unlocked(SubController('unlocked'))

        class RootController(object):
            secret = SecretController()
            notsecret = NotSecretController()
        self.deepsecret_cls = DeepSecretController
        self.secret_cls = SecretController
        self.permissions_checked = permissions_checked
        self.app = TestApp(make_app(RootController(), debug=True, static_root='tests/static'))

    def tearDown(self):
        self.permissions_checked.clear()
        self.secret_cls.authorized = False
        self.deepsecret_cls.authorized = False

    def test_sub_of_both_not_secret(self):
        response = self.app.get('/notsecret/hi/')
        assert response.status_int == 200
        assert response.body == b'Index hi'

    def test_protected_lookup(self):
        response = self.app.get('/secret/hi/', expect_errors=True)
        assert response.status_int == 401
        self.secret_cls.authorized = True
        response = self.app.get('/secret/hi/')
        assert response.status_int == 200
        assert response.body == b'Index hi'
        assert 'secretcontroller' in self.permissions_checked

    def test_secured_notfound_lookup(self):
        response = self.app.get('/secret/notfound/', expect_errors=True)
        assert response.status_int == 404

    def test_secret_through_lookup(self):
        response = self.app.get('/notsecret/hi/deepsecret/', expect_errors=True)
        assert response.status_int == 401

    def test_layered_protection(self):
        response = self.app.get('/secret/hi/deepsecret/', expect_errors=True)
        assert response.status_int == 401
        assert 'secretcontroller' in self.permissions_checked
        self.secret_cls.authorized = True
        response = self.app.get('/secret/hi/deepsecret/', expect_errors=True)
        assert response.status_int == 401
        assert 'secretcontroller' in self.permissions_checked
        assert 'deepsecret' in self.permissions_checked
        self.deepsecret_cls.authorized = True
        response = self.app.get('/secret/hi/deepsecret/')
        assert response.status_int == 200
        assert response.body == b'Deep Secret'
        assert 'secretcontroller' in self.permissions_checked
        assert 'deepsecret' in self.permissions_checked

    def test_cyclical_protection(self):
        self.secret_cls.authorized = True
        self.deepsecret_cls.authorized = True
        response = self.app.get('/secret/1/deepsecret/2/deepsecret/')
        assert response.status_int == 200
        assert response.body == b'Deep Secret'
        assert 'secretcontroller' in self.permissions_checked
        assert 'deepsecret' in self.permissions_checked

    def test_unlocked_lookup(self):
        response = self.app.get('/notsecret/1/deepsecret/2/')
        assert response.status_int == 200
        assert response.body == b'Index 2'
        assert 'deepsecret' not in self.permissions_checked
        response = self.app.get('/notsecret/1/deepsecret/notfound/', expect_errors=True)
        assert response.status_int == 404
        assert 'deepsecret' not in self.permissions_checked

    def test_mixed_protection(self):
        self.secret_cls.authorized = True
        response = self.app.get('/secret/1/deepsecret/notfound/', expect_errors=True)
        assert response.status_int == 404
        assert 'secretcontroller' in self.permissions_checked
        assert 'deepsecret' not in self.permissions_checked

    def test_independent_check_failure(self):
        response = self.app.get('/secret/independent/', expect_errors=True)
        assert response.status_int == 401
        assert len(self.permissions_checked) == 1
        assert 'independent' in self.permissions_checked

    def test_independent_check_success(self):
        self.secret_cls.independent_authorization = True
        response = self.app.get('/secret/independent')
        assert response.status_int == 200
        assert response.body == b'Independent Security'
        assert len(self.permissions_checked) == 1
        assert 'independent' in self.permissions_checked

    def test_wrapped_attribute_failure(self):
        self.secret_cls.independent_authorization = False
        response = self.app.get('/secret/wrapped/', expect_errors=True)
        assert response.status_int == 401
        assert len(self.permissions_checked) == 1
        assert 'independent' in self.permissions_checked

    def test_wrapped_attribute_success(self):
        self.secret_cls.independent_authorization = True
        response = self.app.get('/secret/wrapped/')
        assert response.status_int == 200
        assert response.body == b'Index wrapped'
        assert len(self.permissions_checked) == 1
        assert 'independent' in self.permissions_checked

    def test_lookup_to_wrapped_attribute_on_self(self):
        self.secret_cls.authorized = True
        self.secret_cls.independent_authorization = True
        response = self.app.get('/secret/lookup_wrapped/')
        assert response.status_int == 200
        assert response.body == b'Index wrapped'
        assert len(self.permissions_checked) == 2
        assert 'independent' in self.permissions_checked
        assert 'secretcontroller' in self.permissions_checked

    def test_unlocked_attribute_in_insecure(self):
        response = self.app.get('/notsecret/unlocked/')
        assert response.status_int == 200
        assert response.body == b'Index unlocked'