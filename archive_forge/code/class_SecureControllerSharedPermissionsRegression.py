import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
class SecureControllerSharedPermissionsRegression(PecanTestCase):
    """Regression tests for https://github.com/dreamhost/pecan/issues/131"""

    def setUp(self):
        super(SecureControllerSharedPermissionsRegression, self).setUp()

        class Parent(object):

            @expose()
            def index(self):
                return 'hello'

        class UnsecuredChild(Parent):
            pass

        class SecureChild(Parent, SecureController):

            @classmethod
            def check_permissions(cls):
                return False

        class RootController(object):
            secured = SecureChild()
            unsecured = UnsecuredChild()
        self.app = TestApp(make_app(RootController()))

    def test_inherited_security(self):
        assert self.app.get('/secured/', status=401).status_int == 401
        assert self.app.get('/unsecured/').status_int == 200