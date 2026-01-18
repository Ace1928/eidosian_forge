import sys
import unittest
from webtest import TestApp
from pecan import expose, make_app
from pecan.secure import secure, unlocked, SecureController
from pecan.tests import PecanTestCase
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